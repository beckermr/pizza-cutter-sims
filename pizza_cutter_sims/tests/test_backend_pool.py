import pytest

from schwimmbad import MPIPool, JoblibPool

from pizza_cutter_sims.run_utils import backend_pool


def _double(x):
    return 2*x


@pytest.mark.parametrize("backend", ["loky", "sequential"])
def test_backend_pool(backend):
    with backend_pool(backend) as pool:
        if backend != "mpi":
            assert isinstance(pool, JoblibPool)
        else:
            assert isinstance(pool, MPIPool)

        assert sum(pool.map(_double, range(10))) == sum(map(_double, range(10)))
