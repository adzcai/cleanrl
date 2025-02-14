# Installation

We use the [uv](https://docs.astral.sh/uv/getting-started/installation/) project manager.
Navigate to this directory and run

```bash
uv sync
```

You need to install graphviz to make the MCTS visualizations. On a mac:

```
brew install graphviz
```

You may also need to tell compilers and linkers about its path before running `uv sync`:

```bash
export CFLAGS="-I$(brew --prefix graphviz)/include"
export LDFLAGS="-L$(brew --prefix graphviz)/lib"
```
