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
from collections.abc import Callable, Iterable
from typing import Annotated as Batched
from typing import Any, Literal, Protocol, TypeVar, get_origin

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import rlax
from jaxtyping import Array, Float, Key
from omegaconf import DictConfig, OmegaConf

import wandb
from utils.log_util import dataclass, dict_to_dataclass, typecheck
from wandb.sdk.wandb_run import Run

TConfig = TypeVar("TConfig", bound="Config", contravariant=True)

# warnings.filterwarnings("error")  # turn warnings into errors


@dataclass
class Config:
    """Parent dataclass for configuration options.

    Inherit from this in each algorithm file and add algorithm-specific configurations.
    """

    seed: int
    num_seeds: int
    sweep_method: str  # omegaconf doesn't support Literal["grid", "random", "bayes"]
    command: Literal["run", "sweep", "agent", "jaxpr"] = dc.field(
        metadata={
            "help": "Set to `sweep` to output a wandb sweep configuration yaml instead of executing the run."
            "Set to `agent` to load config from wandb instead of cli."
        },
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
    command="run",
)


@dataclass
class ArchConfig:
    """Network architecture"""

    rnn_size: int
    mlp_size: int
    mlp_depth: int
    goal_dim: int
    activation: str
    projection_kind: Literal["mlp", "cnn", "housemaze"]
    obs_dim: int | None = None
    """Dimension of observation embedding. The number of "orthogonal directions" among observation components."""


@dataclass
class EnvConfig:
    """Arguments to specify an environment."""

    name: str
    source: Literal["gymnax", "brax", "navix", "custom"] = "gymnax"
    kwargs: dict[str, Any] = dc.field(default_factory=dict)


@dataclass
class CollectionConfig:
    """For episode rollouts."""

    total_transitions: int
    num_timesteps: int
    buffer_size_denominator: int | float
    """The time axis of the buffer is int(total_transitions / (num_envs * buffer_size_denominator)).
    i.e. throughout training, the buffer will be filled `buffer_size_denominator` times.
    """
    num_envs: int  # more parallel data collection
    mcts_depth: int
    num_mcts_simulations: int  # stronger policy improvement


@dataclass
class ValueConfig:
    """For parameterizing value function."""

    num_value_bins: int | Literal["scalar"]
    min_value: float
    max_value: float

    @typecheck
    def logits_to_value(
        self,
        value_logits: Float[Array, " *horizon num_value_bins"],
    ) -> Float[Array, " *horizon"]:
        """Convert from logits for two-hot encoding to scalar values."""
        assert self.num_value_bins != "scalar"
        return jnp.asarray(
            rlax.transform_from_2hot(
                jax.nn.softmax(value_logits, axis=-1),
                self.min_value,
                self.max_value,
                self.num_value_bins,
            )
        )

    @typecheck
    def value_to_probs(
        self,
        value: Float[Array, " *horizon"],
    ) -> Float[Array, " *horizon num_value_bins"]:
        """Convert from scalar values to probabilities for two-hot encoding."""
        assert self.num_value_bins != "scalar"
        return jnp.asarray(
            rlax.transform_to_2hot(
                value,
                self.min_value,
                self.max_value,
                self.num_value_bins,
            )
        )


@dataclass
class BootstrapConfig:
    """For bootstrapping with a target network."""

    discount: float
    lambda_gae: float
    target_update_freq: int  # in global iterations
    target_update_size: float
    """The target network is updated `target_update_size` of the way to the online network."""


@dataclass
class OptimConfig:
    """Optimization parameters"""

    num_minibatches: int  # number of gradient descent updates per iteration
    batch_size: int  # reduce gradient variance
    num_timesteps: int
    max_grad_norm: float

    # learning rate
    lr_init: float
    warmup_frac: float
    decay_rate: float
    num_stairs: int
    # end learning rate

    value_coef: float  # scale the value loss
    reward_coef: float  # scale the reward loss
    world_model_gradient_scale: float  # scale the world model gradients per step
    priority_exponent: float  # prioritized replay
    importance_exponent: float
    """Probability to recompute policy targets."""

    def __post_init__(self):
        if self.num_timesteps <= 1:
            raise ValueError("Updates must use at least two timesteps.")


@dataclass
class EvalConfig:
    """Evaluation of the learned policy and value function."""

    warnings: bool
    num_timesteps: int
    num_evals: int
    num_eval_envs: int


@dataclass
class TrainConfig(Config):
    """Training parameters."""

    arch: ArchConfig  # type: ignore
    env: EnvConfig  # type: ignore
    collection: CollectionConfig  # type: ignore
    value: ValueConfig  # type: ignore
    bootstrap: BootstrapConfig  # type: ignore
    optim: OptimConfig  # type: ignore
    eval: EvalConfig  # type: ignore

    @property
    def name(self):
        return f"{self.env.name} {self.collection.total_transitions}"


def get_args(cfg_paths: Iterable[str] = (), cli_args: list[str] | None = None) -> dict:
    assert cfg_paths or cli_args, "No configuration files or CLI arguments provided."
    cfg = OmegaConf.merge(
        # using `structured` prevents addition of ConfigClass fields
        OmegaConf.create(dc.asdict(DEFAULT_CONFIG)),
        *map(OmegaConf.load, cfg_paths),
        OmegaConf.from_cli(cli_args),
    )
    assert OmegaConf.is_dict(cfg), "Configuration must be a dictionary."
    return OmegaConf.to_object(cfg)  # type: ignore


class MakeTrainFn(Protocol[TConfig]):
    def __call__(self, config: TConfig) -> Callable[[Key[Array, ""]], Any]:
        """Returns a jittable `train` function that accepts the merged configuration."""
        raise NotImplementedError


def main(
    ConfigClass: type[TConfig],
    make_train: MakeTrainFn[TConfig],
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
            assert __doc__ is not None, "Docstring is required for help message."
            print(__doc__.format(file=file))
            return
        else:
            ary = cli_args if "=" in arg else cfg_paths
            ary.append(arg)
    cfg_dict = get_args(cfg_paths, cli_args)
    outputs = None

    if cfg_dict["command"] == "agent":
        with wandb.init(config=None) as run:  # set config to None to load from wandb
            outputs = run_train(run, ConfigClass, make_train)

    elif cfg_dict["command"] == "jaxpr":
        cfg = dict_to_dataclass(ConfigClass, cfg_dict)
        train = make_train(cfg)
        jaxpr = jax.make_jaxpr(train)(jr.key(cfg.seed))
        print(jaxpr)

    elif cfg_dict["command"] == "sweep":
        cfg = dict_to_dataclass(ConfigClass, cfg_dict)
        sweep_params, parameters = to_wandb_sweep_parameters(cfg)
        sweep_cfg = {
            "program": file,
            "method": cfg.sweep_method,
            "name": f"{file} sweep {' '.join(sweep_params)}",
            "metric": {"goal": "maximize", "name": "eval/mean_return"},
            "parameters": parameters,
            "command": [
                r"${env}",
                r"${interpreter}",
                r"${program}",
                r"command=agent",
            ],
        }
        sweep_id = wandb.sweep(sweep_cfg)
        print(
            "To launch a SLURM batch job:\n"
            f'SWEEP_ID={sweep_id} sbatch --job-name "{sweep_cfg["name"]}" launch.sh\n'
            "Remember you can also pass sbatch arguments via the command line.\n"
            "Run `man sbatch` for details."
        )

    elif cfg_dict["command"] == "run":
        matplotlib.use("agg")  # enable plotting inside jax callback
        with wandb.init(config=cfg_dict) as run:
            outputs = run_train(run, ConfigClass, make_train)

    else:
        raise ValueError(
            f"Unknown command {cfg_dict['command']}. Use `run`, `sweep`, `agent`, or `jaxpr`."
        )

    if outputs is not None:
        _, mean_eval_reward = outputs
        mean_eval_reward = mean_eval_reward[mean_eval_reward != -jnp.inf]
        print(f"Done training. {mean_eval_reward=}")


def run_train(
    run: Run, ConfigClass: type[TConfig], make_train: MakeTrainFn[TConfig]
) -> Batched[Any, " num_seeds"]:
    # setup config
    cfg = dict_to_dataclass(ConfigClass, wandb.config)  # type: ignore
    run.name = cfg.name
    train = make_train(cfg)
    keys = jr.split(jr.key(cfg.seed), cfg.num_seeds)

    # call train
    # with jax.profiler.trace(f"/tmp/{os.environ['WANDB_PROJECT']}-trace", create_perfetto_link=True):
    outputs = jax.jit(jax.vmap(train))(keys)
    return jax.block_until_ready(outputs)


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
