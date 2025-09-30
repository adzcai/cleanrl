import jax
import pytest

from ilx.core.maps import LARGER_MAP, SIMPLE_MAP
from ilx.core.mdp import GridEnv

jax.config.update("jax_disable_jit", True)


@pytest.fixture
def simple_env():
    return GridEnv(SIMPLE_MAP, 0.99)


@pytest.fixture
def larger_env():
    return GridEnv(LARGER_MAP, 0.99)


@pytest.fixture(params=[SIMPLE_MAP, LARGER_MAP])
def env(request):
    return GridEnv(request.param, 0.99)
