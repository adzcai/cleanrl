"""See below for arguments and usage examples.

RUNNING
-------
To run the algorithm,
create a yaml file matching the required config and run

    python {file} $CONFIG_YAML total_transitions=100000  # other parameters etc

You can also pass multiple yamls to be merged (later ones take precedence)

    python {file} train.yaml debug.yaml

WANDB
-----
We do not provide explicit commands for configuring wandb.
You can control wandb using its environment variables (https://docs.wandb.ai/guides/track/environment-variables/)
or by running

    wandb init --project $PROJECT --entity $ENTITY

before launching any sweeps or agents.
For example, you can set `WANDB_RUN_GROUP` to categorize the run into a group.

SWEEPS
------
To run a wandb sweep, first execute

    python {file} $CONFIG_YAML sweep=True

to start a new wandb sweep. This will output a $SWEEP_ID. Then run

    wandb agent $SWEEP_ID --count $COUNT

to run one or more agents.

LIBRARY USAGE
-------------
This file defines a `main` function that is meant to be imported into other files.
For example, in your algorithm file,

    from config import Config, main

    @dataclass
    class TrainConfig(Config):
        ...

    def make_train(config: TrainConfig):
        ...

    if __name__ == "__main__":
        main(TrainConfig, make_train, Path(__file__).name)
"""

import dataclasses as dc
import sys
from collections.abc import Callable
from typing import Any, Literal, TypeVar, get_origin

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key
from omegaconf import OmegaConf

import wandb
from log_util import dataclass, dict_to_dataclass

TConfig = TypeVar("TConfig", bound="Config")

# warnings.filterwarnings("error")  # turn warnings into errors


@dataclass
class Config:
    """Parent dataclass for configuration options.

    Inherit from this in each algorithm file and add algorithm-specific configurations.
    """

    seed: int
    num_seeds: int
    sweep_method: str  # omegaconf doesn't support Literal["grid", "random", "bayes"]
    sweep: bool = dc.field(
        metadata={"help": "Output a wandb sweep configuration yaml instead of executing the run."},
    )
    agent: bool = dc.field(
        metadata={"help": "Load config from wandb instead of cli."},
    )

    @property
    def name(self):
        """The wandb run name. Can be overridden."""
        raise NotImplementedError


# outside the class to allow subclasses to have arguments without defaults
DEFAULT_CONFIG = Config(
    seed=0,
    num_seeds=1,
    sweep_method="random",
    sweep=False,
    agent=False,
)


@dataclass
class ArchConfig:
    """Network architecture"""

    kind: Literal["mlp", "cnn"]
    rnn_size: int
    mlp_size: int
    mlp_depth: int
    activation: str

    # to be replaced
    num_actions: int = -1
    num_goals: int = -1
    num_value_bins: int | Literal["scalar"] = -1


@dataclass
class EnvConfig:
    """Arguments to specify an environment."""

    env_name: str
    horizon: int
    env_source: Literal["gymnax", "brax", "navix", "custom"] = "gymnax"
    env_kwargs: dict[str, Any] = dc.field(default_factory=dict)


@dataclass
class CollectionConfig:
    """For episode rollouts."""

    total_transitions: int
    num_envs: int  # more parallel data collection
    mcts_depth: int
    num_mcts_simulations: int  # stronger policy improvement


@dataclass
class ValueConfig:
    """For parameterizing value function."""

    num_value_bins: int | Literal["scalar"]
    min_value: float
    max_value: float


@dataclass
class BootstrapConfig:
    """For bootstrapping with a target network."""

    discount: float
    lambda_gae: float
    target_update_freq: int  # in global iterations
    target_update_size: float


@dataclass
class OptimConfig:
    """Optimization parameters"""

    num_minibatches: int  # number of gradient descent updates per iteration
    batch_size: int  # reduce gradient variance
    lr_init: float
    max_grad_norm: float
    value_coef: float  # scale the value loss
    reward_coef: float  # scale the reward loss
    priority_exponent: float  # prioritized replay


@dataclass
class EvalConfig:
    """Evaluation of the learned policy and value function."""

    warnings: bool
    eval_horizon: int
    num_evals: int
    num_eval_envs: int


@dataclass
class TrainConfig(Config):
    """Training parameters."""

    arch: ArchConfig
    env: EnvConfig
    collection: CollectionConfig
    value: ValueConfig
    bootstrap: BootstrapConfig
    optim: OptimConfig
    eval: EvalConfig

    @property
    def name(self):
        return f"{self.env.env_name} {self.collection.total_transitions}"


def main(
    ConfigClass: type[TConfig],
    make_train: Callable[[TConfig], Callable[[Key[Array, ""]], Any]],
    file: str,
) -> None:
    """Merge configurations from yaml files and cli and pass it to `make_train`.

    Args:
        ConfigClass (type[T]): The dataclass describing the configuration options.
        make_train (Callable[[T], Callable[[Key, ()], Any]]): Returns a jittable `train` function that accepts the merged configuration.
        file (str): The file that `main` is being called from (for generating sweep config).
    """
    # merge configuration
    cfg_paths, cli_args = [], []
    for arg in sys.argv[1:]:
        if arg in ["-h", "--help"]:
            print(__doc__.format(file=file))
            sys.exit(0)
        (cli_args if "=" in arg else cfg_paths).append(arg)
    cfg: TConfig = OmegaConf.merge(
        # using `structured` prevents addition of ConfigClass fields
        OmegaConf.create(dc.asdict(DEFAULT_CONFIG)),
        *map(OmegaConf.load, cfg_paths),
        OmegaConf.from_cli(cli_args),
    )

    if cfg.sweep:
        cfg = dict_to_dataclass(ConfigClass, cfg)
        sweep_cfg = as_sweep_config(cfg, file)
        sweep_id = wandb.sweep(sweep_cfg)
        print(
            "To launch a SLURM batch job:\n"
            f"SWEEP_ID={sweep_id} sbatch --job-name {sweep_cfg['name']} launch.sh\n"
            "Remember you can also pass sbatch arguments via the command line.\n"
            "Run `man sbatch` for details."
        )
    else:
        with wandb.init(config=None if cfg.agent else OmegaConf.to_object(cfg)) as run:
            cfg = dict_to_dataclass(ConfigClass, wandb.config)
            run.name = cfg.name
            train = make_train(cfg)
            keys = jr.split(jr.key(cfg.seed), cfg.num_seeds)
            # with jax.profiler.trace(f"/tmp/{os.environ['WANDB_PROJECT']}-trace", create_perfetto_link=True):
            outputs = jax.jit(jax.vmap(train))(keys)
            _, mean_eval_reward = jax.block_until_ready(outputs)
            mean_eval_reward = mean_eval_reward[mean_eval_reward != -jnp.inf]
        print(f"Done training. {mean_eval_reward=}")


def as_sweep_config(config: Config, file: str) -> dict:
    """Generates the wandb sweep config. Does not upload to wandb."""
    sweep_params, parameters = to_wandb_sweep_parameters(config)
    return {
        "program": file,
        "method": config.sweep_method,
        "name": f"{file} sweep {' '.join(sweep_params)}",
        "metric": {"goal": "maximize", "name": "eval/mean_return"},
        "parameters": parameters,
        "command": [
            r"${env}",
            r"${interpreter}",
            r"${program}",
            r"agent=True",
        ],
    }


def to_wandb_sweep_parameters(config: Config) -> tuple[set[str], dict]:
    """Turn a config dataclass instance to a wandb sweep parameters dictionary.

    Args:
        config (Config): The config to transform

    Returns:
        tuple[set[str], dict]: The parameters being swept and the sweep configuration dictionary.
    """
    sweep_params, parameters = set(), dict()
    for field in dc.fields(config):
        value = getattr(config, field.name)
        if dc.is_dataclass(field.type):
            swept, params = to_wandb_sweep_parameters(value)
            sweep_params |= swept
            value = dict(parameters=params)
        elif isinstance(value, dict) and get_origin(field.type) is not dict:
            # swept parameter
            sweep_params.add(field.name)
        else:
            value = dict(value=value)
        parameters[field.name] = value
    return sweep_params, parameters
