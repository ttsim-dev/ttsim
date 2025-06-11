import numpy
import pytest


# content of conftest.py
def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default="numpy",
        help="The backend to test against (e.g., --backend=numpy --backend=jax)",
    )


@pytest.fixture
def backend_xnp(request):
    backend = request.config.getoption("--backend")
    if backend == "numpy":
        xnp = numpy
    else:
        import jax

        xnp = jax.numpy
    return backend, xnp
