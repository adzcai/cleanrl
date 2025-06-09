import pytest

from testing.dummy_env import make_dummy_env
from testing.pytree_env import make_pytree_env


@pytest.fixture
def dummy_env():
    return make_dummy_env()


@pytest.fixture
def pytree_env():
    return make_pytree_env()
