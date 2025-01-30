# Installation

You need to install graphviz first. On a mac:

```
brew install graphviz
```

You also need to specify the path:

```bash
export CFLAGS="-I$(brew --prefix graphviz)/include"
export LDFLAGS="-L$(brew --prefix graphviz)/lib"
```

Then you can install the package with `uv`:

```bash
uv sync
```
