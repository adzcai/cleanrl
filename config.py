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
import json
import yaml

# typing
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, TypeVar, get_args, get_origin
from jaxtyping import Key, Array

# util
import argparse
from pathlib import Path
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

    python {file} $CONFIG_YAML

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

    python {file} $CONFIG_YAML --sweep

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
    config_path: tuple[str, ...] = dc.field(default=(), metadata={"nargs": "*"})
    sweep_method: Literal["grid", "random", "bayes"] = "random"
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


def get_cli_args(ConfigClass: type[Config], file: str) -> dict:
    """Read the configuration specified by the configuration dataclass."""
    # construct parser. none of the arguments have default values to allow file composition
    parser = argparse.ArgumentParser(usage=CLI_DESCRIPTION.format(file=file))

    # add the ConfigClass to the parser (in groups)
    for field in dc.fields(ConfigClass):
        group_parser = parser.add_argument_group(field.type.__name__)
        for subfield in dc.fields(field.type):
            add_field_to_parser(group_parser, subfield)
    args = parser.parse_args()

    # read hierarchical structure into cfg_dict
    cfg_dict = {}

    # read from config files
    for cfg_path in map(Path, args.config_path):
        with cfg_path.open() as f:
            if cfg_path.suffix == ".yaml":
                cfg: dict = yaml.safe_load(f)
            elif cfg_path.suffix == ".json":
                cfg: dict = json.load(f)
            else:
                raise ValueError("Only JSON and YAML config files supported. Got " + cfg_path)
        extend_dict(cfg_dict, cfg)

    # read from cli
    extend_dict(cfg_dict, unflatten(ConfigClass, args))

    return cfg_dict


def extend_dict(a: dict, b: dict) -> None:
    """Deeply overwrite a with b."""
    for key, value in b.items():
        if isinstance(value, dict):
            a.setdefault(key, {})
            extend_dict(a[key], value)
        else:
            a[key] = value


def unflatten(cls: type[T], args: argparse.Namespace) -> dict:
    out = {}
    for field in dc.fields(cls):
        if dc.is_dataclass(field.type):
            unflatten(field.type, args)
        out


def add_field_to_parser(parser: argparse.ArgumentParser, field: dc.Field) -> None:
    """Add a dataclass field to the cli argument parser based on its type.

    Literal fields get translated to choices.
    bools get translated to flags.
    variable arguments get the proper type assigned.
    """
    nargs = field.metadata.get("nargs", None)
    help = field.metadata.get("help", None)
    if get_origin(field.type) is Literal:
        parser.add_argument(
            "--" + field.name,
            type=str,
            choices=get_args(field.type),
            nargs=nargs,
            help=help,
        )
    elif field.type is bool:
        parser.add_argument(
            "--" + field.name,
            # don't set default
            action="store_const",
            const=True,
            help=help,
        )
    elif nargs in ["*", "+"]:
        assert get_origin(field.type) is tuple
        # assume tuple[tp, ...]
        tp, _ = get_args(field.type)
        # make config_path positional
        parser.add_argument(
            field.name,
            type=tp,
            nargs=nargs,
            help=help,
        )
    else:
        parser.add_argument(
            "--" + field.name,
            type=field.type if callable(field.type) and nargs is None else None,
            nargs=nargs,
            help=help,
        )


def dict_to_dataclass(cls: type[T], obj: dict) -> T:
    """Deeply convert a structured dictionary to the corresponding dataclass instance."""
    out = {}
    for field in dc.fields(cls):
        if field.name not in obj:
            raise ValueError(f"Field {field.name} missing when constructing {cls}")
        value = obj[field.name]
        if dc.is_dataclass(field.type):
            value = dict_to_dataclass(field.type, value)
        out[field.name] = value
    return cls(**out)


def to_parameters(config: Config) -> tuple[set[str], dict]:
    """Turn a config dataclass instance to a wandb sweep parameters dictionary."""
    sweep_params = set()
    parameters = {}
    for field in dc.fields(config):
        value = getattr(config, field.name)
        if dc.is_dataclass(field.type):
            swept, params = to_parameters(value)
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
    sweep_params, parameters = to_parameters(config)

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
            r"--agent",
        ],
    }


def main(
    ConfigClass: type[T],
    make_train: Callable[[T], Callable[[Key[Array, ""]], Any]],
    file: str,
) -> None:
    """Merge configurations from yaml files and cli and pass it to `make_train`.

    Args:
        ConfigClass (type[T]): The dataclass describing the configuration options.
        make_train (Callable[[T], Callable[[Key, ()], Any]]): Returns a jittable `train` function that accepts the merged configuration.
        file (str): The file that `main` is being called from (for generating sweep config).
    """
    args = get_cli_args(ConfigClass, file)
    experiment = ExperimentConfig(**args["experiment"])
    if experiment.sweep:
        sweep_config = dict_to_dataclass(ConfigClass, args).as_sweep_config(file)
        yaml.safe_dump(sweep_config, sys.stdout)
        wandb.sweep(sweep_config)
    else:
        with wandb.init(config=None if experiment.agent else args):
            del experiment
            config = dict_to_dataclass(ConfigClass, wandb.config.as_dict())
            train = make_train(config)
            output = jax.jit(train)(jr.key(config.experiment.seed))
            # with jax.profiler.trace(f"/tmp/{WANDB_PROJECT}-trace", create_perfetto_link=True):
            _, mean_eval_reward = jax.block_until_ready(output)
            mean_eval_reward = mean_eval_reward[mean_eval_reward != -jnp.inf]
        print(f"Done training. {mean_eval_reward=}")
