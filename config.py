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

    @dataclass(frozen=True)
    class TrainConfig(Config):
        ...

    def make_train(config: TrainConfig):
        ...

    if __name__ == "__main__":
        main(TrainConfig, make_train, Path(__file__).name)
"""

# jax
import jax
import jax.random as jr
import jax.numpy as jnp

# logging
import sys
import wandb

# typing
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar
from jaxtyping import Key, Array

# util
from omegaconf import OmegaConf
import dataclasses as dc

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

T = TypeVar("T")
TConfig = TypeVar("TConfig", bound="Config")


@dataclass(frozen=True)
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


# outside the class to allow subclasses to have arguments without defaults
DEFAULT_CONFIG = Config(
    seed=0,
    num_seeds=1,
    sweep_method="random",
    sweep=False,
    agent=False,
)


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
            sweep_params.add(field.name)
        else:
            value = dict(value=value)
        parameters[field.name] = value
    return sweep_params, parameters


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


def dict_to_dataclass(cls: type[T], obj: dict) -> T:
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
        if field.name not in obj and field.default == dc.MISSING:
            raise ValueError(f"Field {field.name} missing when constructing {cls}")
        value = obj[field.name]
        if dc.is_dataclass(field.type):
            value = dict_to_dataclass(field.type, value)
        out[field.name] = value
    return cls(**out)


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
        wandb.sweep(as_sweep_config(cfg, file))
    else:
        with wandb.init(config=None if cfg.agent else OmegaConf.to_object(cfg)):
            cfg = dict_to_dataclass(ConfigClass, wandb.config)
            train = make_train(cfg)
            keys = jr.split(jr.key(cfg.seed), cfg.num_seeds)
            # with jax.profiler.trace(f"/tmp/{os.environ['WANDB_PROJECT']}-trace", create_perfetto_link=True):
            outputs = jax.jit(jax.vmap(train))(keys)
            _, mean_eval_reward = jax.block_until_ready(outputs)
            mean_eval_reward = mean_eval_reward[mean_eval_reward != -jnp.inf]
        print(f"Done training. {mean_eval_reward=}")
