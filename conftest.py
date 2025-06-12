import pytest

from ttsim.interface_dag_elements.backend import dnp as ttsim_dnp
from ttsim.interface_dag_elements.backend import xnp as ttsim_xnp


# content of conftest.py
def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default="numpy",
        help="The backend to test against (e.g., --backend=numpy --backend=jax)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "skipif_jax: skip test if backend is jax")


@pytest.fixture
def backend(request):
    backend = request.config.getoption("--backend")
    return backend


@pytest.fixture
def xnp(request):
    backend = request.config.getoption("--backend")
    return ttsim_xnp(backend)


@pytest.fixture
def dnp(request):
    backend = request.config.getoption("--backend")
    return ttsim_dnp(backend)


@pytest.fixture(autouse=True)
def skipif_jax(request, backend):
    """Automatically skip tests marked with skipif_jax when backend is jax."""
    if request.node.get_closest_marker("skipif_jax") and backend == "jax":
        pytest.skip("Cannot run this test with Jax")
