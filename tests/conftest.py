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
