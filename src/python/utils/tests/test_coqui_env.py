from mpi4py import MPI
import pytest

import coqui


@pytest.fixture(scope="session")
def mpi():
    """Fixture that returns a CoQui MPI handler usable across all tests."""
    mpi_context = coqui.MpiHandler()
    coqui.set_verbosity(mpi_context, output_level=2)
    return mpi_context


def test_mpi_handler(mpi):
    assert mpi.comm_rank() == MPI.COMM_WORLD.Get_rank()
    assert mpi.comm_size() == MPI.COMM_WORLD.Get_size()
    assert mpi.internode_size() >= 1
    assert mpi.intranode_size() >= 1

    if mpi.root():
        print(mpi)


def test_verbosity(mpi):
    try:
        # Set verbosity levels to 0
        coqui.set_verbosity(mpi, output_level=0)
        # Update the verbosity level to 4
        coqui.set_verbosity(mpi, output_level=2)
    except Exception as e:
        pytest.fail(f"set_verbosity raised an exception: {e}")
