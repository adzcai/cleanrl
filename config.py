"""Boilerplate for running jobs and merging configs.

Good old argparse does the job.
OmegaConf or hydra feels like overkill.

This file defines `main`,
which is meant to be imported into other files.
For example, in `muzero.py`,

    from config import Config, main

    @dataclass(frozen=True)
    class TrainConfig(Config):
        ...

    if __name__ == "__main__":
        main(TrainConfig, make_train, Path(__file__).name)"""

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

CLI_DESCRIPTION = """See below for arguments and usage examples.

RUNNING
-------
To run the algorithm,
create a yaml file matching the required config and run

    python {file} $CONFIG_YAML collection.total_transitions=100000  # etc

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

    python {file} $CONFIG_YAML experiment.sweep=True

to start a new wandb sweep.
This will output a $SWEEP_ID. Then run

    wandb agent $SWEEP_ID --count $COUNT

to run one or more agents."""

T = TypeVar("T")
TConfig = TypeVar("TConfig", bound="Config")


@dataclass(frozen=True)
class ExperimentConfig:
    """Command-line arguments."""

    seed: int = 0
    num_seeds: int = 1
    sweep_method: str = "random"  # omegaconf doesn't support Literal["grid", "random", "bayes"]
    sweep: bool = dc.field(
        default=False,
        metadata={"help": "Output a wandb sweep configuration yaml instead of executing the run."},
    )
    agent: bool = dc.field(
        default=False,
        metadata={"help": "Load config from wandb instead of cli."},
    )


@dataclass(frozen=True)
class Config:
    """Parent dataclass for configuration options.

    Inherit from this in each algorithm file and add algorithm-specific configurations.
    """

    experiment: ExperimentConfig


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
        "method": config.experiment.sweep_method,
        "name": f"sweep {' '.join(sweep_params)}",
        "metric": {"goal": "maximize", "name": "eval/mean_return"},
        "parameters": parameters,
        "command": [
            r"${env}",
            r"${interpreter}",
            r"${program}",
            r"experiment.agent=True",
        ],
    }


def dict_to_dataclass(cls: type[T], obj: dict) -> T:
    """Deeply convert a structured dictionary to the corresponding dataclass instance."""
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
            print(CLI_DESCRIPTION.format(file=file))
            sys.exit(0)
        (cli_args if "=" in arg else cfg_paths).append(arg)
    cfg: TConfig = OmegaConf.merge(
        OmegaConf.structured(Config(experiment=ExperimentConfig())),
        *map(OmegaConf.load, cfg_paths),
        OmegaConf.from_cli(cli_args),
    )

    if cfg.experiment.sweep:
        wandb.sweep(as_sweep_config(cfg, file))
    else:
        with wandb.init(config=None if cfg.experiment.agent else OmegaConf.to_object(cfg)):
            cfg = dict_to_dataclass(ConfigClass, wandb.config)
            train = make_train(cfg)
            keys = jr.split(jr.key(cfg.experiment.seed), cfg.experiment.num_seeds)
            # with jax.profiler.trace(f"/tmp/{os.environ['WANDB_PROJECT']}-trace", create_perfetto_link=True):
            outputs = jax.jit(jax.vmap(train))(keys)
            _, mean_eval_reward = jax.block_until_ready(outputs)
            mean_eval_reward = mean_eval_reward[mean_eval_reward != -jnp.inf]
        print(f"Done training. {mean_eval_reward=}")
