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
import os
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Annotated as Batched
from typing import Any, Literal, Protocol, TypeVar, get_origin

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import rlax
import wandb
from jaxtyping import Array, Float, Key
from omegaconf import DictConfig, ListConfig, OmegaConf
from wandb.sdk.wandb_run import Run

from cleanrl_utils.jax_utils import BootstrapConfig
from cleanrl_utils.log_utils import typecheck
from utils.prioritized_buffer import BufferConfig
from utils.structures import TDataclass, dataclass

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
    mode: Literal["run", "sweep", "agent", "jaxpr"] = dc.field(
        metadata={
            "help": "Set to `sweep` to output a wandb sweep configuration yaml instead of executing the run."
            "Set to `agent` to load config from wandb instead of cli."
        },
    )

    @property
    def name(self):
        """The wandb run name. Can be overridden."""
        raise NotImplementedError

    def validate(self):
        """Validate the configuration.

        This is called before training starts.
        Override this in subclasses to add custom validation logic.
        """
        raise NotImplementedError("You must override the `validate` method in your Config subclass.")


# outside the class to allow subclasses to have arguments without defaults
DEFAULT_CONFIG = Config(
    seed=0,
    num_seeds=1,
    sweep_method="random",
    mode="run",
)


@dataclass
class ArchConfig:
    """Network architecture"""

    dyn_size: int
    mlp_size: int
    mlp_depth: int
    goal_dim: int
    activation: str
    projection_kind: Literal["mlp", "cnn", "housemaze"]
    world_model_gradient_scale: float  # scale the world model gradients per step
    obs_dim: int | None = None
    """Dimension of observation embedding.
    The number of "orthogonal directions" among observation components."""


@dataclass
class EnvConfig:
    """Arguments to specify an environment."""

    name: str
    source: Literal["gymnax", "brax", "navix", "custom"] = "gymnax"
    kwargs: dict[str, Any] = dc.field(default_factory=dict)


@dataclass
class ValueConfig:
    """For parameterizing value function."""

    num_value_bins: int
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
class OptimConfig:
    """Optimization parameters"""

    total_updates: int  # total number of gradient descent updates
    num_updates_per_iter: int  # number of gradient descent updates per iteration
    batch_size: int  # reduce gradient variance
    target_update_freq: int  # in global iterations

    @property
    def num_iters(self) -> int:
        return self.total_updates // self.num_updates_per_iter


@dataclass
class ReplayConfig:
    """Replay buffer configuration."""

    priority_exponent: float  # prioritized replay
    importance_exponent: float
    """Probability to recompute policy targets."""


@dataclass
class LossConfig:
    """Loss coefficients for the different components of the loss function."""

    policy_coef: float  # scale the policy loss
    value_coef: float  # scale the value loss
    reward_coef: float  # scale the reward loss


@dataclass
class EvalConfig:
    """Evaluation of the learned policy and value function."""

    warnings: bool
    eval_freq: int


@dataclass
class TrainConfig(Config):
    """Training parameters."""

    arch: ArchConfig  # type: ignore
    env: EnvConfig  # type: ignore
    value: ValueConfig  # type: ignore
    bootstrap: BootstrapConfig  # type: ignore
    optim: OptimConfig  # type: ignore
    buffer: BufferConfig  # type: ignore
    replay: ReplayConfig  # type: ignore
    mcts: dict[str, Any]  # type: ignore
    lr: dict[str, Any]  # type: ignore
    loss: LossConfig  # type: ignore
    eval: EvalConfig  # type: ignore

    @property
    def name(self):
        return f"{self.env.name} {self.optim.total_updates}"

    def validate(self):
        if self.buffer.sample_length <= 1:
            raise ValueError("Updates must use at least two timesteps for bootstrapping.")


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


def dict_to_dataclass(cls: type[TDataclass], obj: Mapping[str, Any]) -> TDataclass:
    """Cast a dictionary to a dataclass instance.

    Args:
        cls (type[T]): The dataclass to cast to.
        obj (dict): The dictionary matching the dataclass fields.

    Raises:
        ValueError: If any required arguments are missing.

    Returns:
        T: The dataclass instance.
    """
    out = {}
    for field in dc.fields(cls):
        if field.name in obj:
            value = obj[field.name]
        elif field.default is not dc.MISSING:
            value = field.default
        elif field.default_factory is not dc.MISSING:
            value = field.default_factory()
        else:
            raise ValueError(f"Field {field.name} missing when constructing {cls}")
        if dc.is_dataclass(tp := field.type):
            value = dict_to_dataclass(tp, value)  # type: ignore
        if isinstance(value, (DictConfig, ListConfig)):
            value = OmegaConf.to_object(value)
        out[field.name] = value
    return cls(**out)


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

    mode = cfg_dict["mode"]

    if mode == "agent":
        with wandb.init(config=None) as run:  # set config to None to load from wandb
            run_train(run, ConfigClass, make_train)

    elif mode == "jaxpr":
        cfg = dict_to_dataclass(ConfigClass, cfg_dict)
        train = make_train(cfg)
        jaxpr = jax.make_jaxpr(train)(jr.key(cfg.seed))
        print(jaxpr)

    elif mode == "sweep":
        cfg = dict_to_dataclass(ConfigClass, cfg_dict)
        sweep_params, parameters = to_wandb_sweep_parameters(cfg)

        sweep_cfg = {
            "program": file,
            "method": cfg.sweep_method,
            "name": f"{Path(file).name} sweep {' '.join(sweep_params)}",
            "metric": {"goal": "minimize", "name": "train/td_error"},
            "parameters": parameters,
            "command": [
                r"${env}",
                r"${interpreter}",
                r"${program}",
                r"mode=agent",
            ],
        }
        wandb_project = os.environ.get("WANDB_PROJECT")
        sweep_id = wandb.sweep(sweep_cfg, project=wandb_project)
        print(
            "To launch a SLURM batch job:\n"
            "\n"
            f"WANDB_PROJECT={wandb_project} SWEEP_ID={sweep_id} sbatch \\\n"
            f'  --job-name "{sweep_cfg["name"]}" \\\n'
            "  --gpus 2 \\\n"
            "  --time 0-00:20 \\\n"
            "  --mem 10GB \\\n"
            "  launch.sh\n"
            "\n"
            "Or run\n"
            "\n"
            f"wandb agent {sweep_id} --count 2\n"
            "\n"
            "to run the sweep locally.\n"
            "Remember you can also pass sbatch arguments via the command line.\n"
            "Run `man sbatch` for details."
        )

    elif mode == "run":
        matplotlib.use("agg")  # enable plotting inside jax callback
        with wandb.init(config=cfg_dict) as run:
            run_train(run, ConfigClass, make_train)
            time.sleep(2)  # wait for wandb to sync logs

    else:
        raise ValueError(f"Unknown command {mode}. Use `run`, `sweep`, `agent`, or `jaxpr`.")

    print("Done training!")


def run_train(run: Run, ConfigClass: type[TConfig], make_train: MakeTrainFn[TConfig]) -> Batched[Any, " num_seeds"]:
    # setup config
    cfg = dict_to_dataclass(ConfigClass, wandb.config)  # type: ignore
    cfg.validate()
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
        elif isinstance(value, dict):
            if get_origin(field.type) is dict:
                sweep_params |= {k for k, v in value.items() if isinstance(v, dict)}
                value = {k: v if isinstance(v, dict) else dict(value=v) for k, v in value.items()}
                value = dict(parameters=value)
            else:
                # swept parameter
                sweep_params.add(field.name)
        else:
            value = dict(value=value)
        parameters[field.name] = value

    return sweep_params, parameters
