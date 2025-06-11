import pytest

from envs.dummy_env import make_dummy_env
from envs.pytree_env import make_pytree_env


@pytest.fixture(params=(1, 2, 5))
def dummy_env_params(request):
    return make_dummy_env(max_horizon=request.param)


@pytest.fixture
def pytree_env():
    return make_pytree_env()
