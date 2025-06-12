# jaxrl

Some deep RL experiments in Jax.

## Installation

We use the [uv](https://docs.astral.sh/uv/getting-started/installation/) project manager.
Navigate to this directory and run

```bash
uv sync
```

You need to install graphviz to make the MCTS visualizations. On Mac,

```bash
brew install graphviz
```

You may also need to tell compilers and linkers about its path before running `uv sync`:

```bash
export CFLAGS="-I$(brew --prefix graphviz)/include"
export LDFLAGS="-L$(brew --prefix graphviz)/lib"
```

## Running

```bash
uv run muzero -- src/config/catch.yaml
```

## Batch jobs

You can run batch jobs by setting

```bash
python src/experiments/muzero.py src/config/catch.yaml src/config/catch_sweep.yaml mode=sweep
```

This will output further instructions for executing the batch job.
