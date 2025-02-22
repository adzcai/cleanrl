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
from collections import defaultdict
from pathlib import Path
import dataclasses as dc

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass


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
        metadata={
            "help": "Flag that indicates whether the script is being run as an agent. "
            "Loads config from wandb instead of cli."
        },
    )


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig

    @classmethod
    def from_dict(cls, args: dict):
        """Construct a TrainConfig from the merged cli argument dictionary.

        Args:
            args (dict): The nested dictionary of cli args.

        Returns:
            TrainConfig: The args packaged into a TrainConfig dataclass.
        """
        config = {}
        for field in dc.fields(cls):
            if field.name not in args:
                raise ValueError(f"Missing field {field.name}")
            subcfg = args[field.name]
            missing = {
                subfield.name
                for subfield in dc.fields(field.type)
                if subfield.default == dc.MISSING and subfield.name not in subcfg
            }
            if missing:
                raise ValueError(f"Missing subfields for {field.name}: {', '.join(missing)}")
            config[field.name] = field.type(**subcfg)
        config = cls(**config)
        config.validate()
        return config

    def validate(self) -> None:
        for field in dc.fields(self):
            subcfg = getattr(self, field.name)
            for subfield in dc.fields(field.type):
                value = getattr(subcfg, subfield.name)
                if isinstance(value, dict) and not self.experiment.sweep:
                    raise ValueError(
                        "Dictionary values should only be passed when generating a sweep configuration with --generate_sweep. "
                        f"Found {value} at {field.name}.{subfield.name}"
                    )

    def as_sweep_config(self, file: str) -> dict:
        """Generates the wandb sweep config. Does not upload to wandb."""
        sweep_params = set()
        parameters = {}
        for field in dc.fields(self):
            subcfg = getattr(self, field.name)
            obj = {}
            for subfield in dc.fields(field.type):
                value = getattr(subcfg, subfield.name)
                if isinstance(value, dict):
                    sweep_params.add(subfield.name)
                    obj[subfield.name] = value
                else:
                    obj[subfield.name] = {"value": value}
            parameters[field.name] = dict(parameters=obj)  # wandb nested parameters

        return {
            "program": file,
            "method": self.experiment.sweep_method,
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


def get_cli_args(ConfigClass: type[Config], file: str) -> dict:
    """Command line interface. Returns `None` if the `--agent` flag is passed."""
    # construct parser. none of the arguments have default values to allow file composition
    parser = argparse.ArgumentParser(usage=CLI_DESCRIPTION.format(file=file))

    # add the ConfigClass to the parser (in groups)
    for field in dc.fields(ConfigClass):
        group_parser = parser.add_argument_group(field.type.__name__)
        for subfield in dc.fields(field.type):
            add_field_to_parser(group_parser, subfield)
    args = parser.parse_args()

    # read hierarchical structure into cfg_dict
    cfg_dict = defaultdict(dict)

    # read from config files
    for cfg_path in map(Path, args.config_path):
        with cfg_path.open() as f:
            if cfg_path.suffix == ".yaml":
                cfg: dict = yaml.safe_load(f)
            elif cfg_path.suffix == ".json":
                cfg = json.load(f)
            else:
                raise ValueError("Only JSON and YAML config files supported. Got " + cfg_path)
        for k, subcfg in cfg.items():
            cfg_dict[k] |= subcfg

    # read from cli
    for field in dc.fields(ConfigClass):
        cfg_dict[field.name] |= {
            subfield.name: subcfg
            for subfield in dc.fields(field.type)
            if (subcfg := getattr(args, subfield.name, None)) is not None
        }

    return cfg_dict


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


T = TypeVar("T", bound=Config)


def main(
    ConfigClass: type[T],
    make_train: Callable[[T], Callable[[Key[Array, ""]], Any]],
    file: str,
):
    args = get_cli_args(ConfigClass, file)
    experiment = ExperimentConfig(**args["experiment"])
    if experiment.sweep:
        sweep_config = ConfigClass.from_dict(args).as_sweep_config(file)
        yaml.safe_dump(sweep_config, sys.stdout)
        wandb.sweep(sweep_config)
    else:
        with wandb.init(config=None if experiment.agent else args):
            del experiment
            config = ConfigClass.from_dict(wandb.config.as_dict())
            train = make_train(config)
            output = jax.jit(train)(jr.key(config.experiment.seed))
            # with jax.profiler.trace(f"/tmp/{WANDB_PROJECT}-trace", create_perfetto_link=True):
            _, mean_eval_reward = jax.block_until_ready(output)
            mean_eval_reward = mean_eval_reward[mean_eval_reward != -jnp.inf]
        print(f"Done training. {mean_eval_reward=}")
