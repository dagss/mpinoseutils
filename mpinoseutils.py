from __future__ import print_function
import sys
import contextlib
import functools

from mpi4py import MPI
import numpy as np

__all__ = ['wait_for_turn', 'mprint', 'meq_', 'assert_eq_across_ranks',
           'mpitest']
           
        

#
# Printing routines
#

@contextlib.contextmanager
def wait_for_turn(comm):
    assert isinstance(comm, MPI.Comm)
    rank = comm.Get_rank()
    exc_info = None
    for i in range(comm.Get_size()):
        if i == rank:
            try:
                yield
            except:
                exc_info = sys.exc_info()
        comm.Barrier()
    # Raise any exception
    if exc_info is not None and comm.Get_rank() == 0:
        raise exc_info[0], exc_info[1], exc_info[2]

def mprint(comm, *args):
    assert isinstance(comm, MPI.Comm)
    with wait_for_turn(comm):
        print('%d:' % comm.Get_rank(), *args)

# TODO: mpprint

#
# Assertions
#

def meq_(comm, expected, got):
    assert type(expected) is list and len(expected) == comm.Get_size()
    rank = comm.Get_rank()
    if got != expected[rank]:
        raise AssertionError("Rank %d: Expected '%r' but got '%r'" % (
            rank, expected[rank], got))



def assert_eq_across_ranks(comm, x):
    lst = comm.gather(x, root=0)
    if comm.Get_rank() == 0:
        for i in range(1, len(lst)):
            y = lst[i]
            if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                is_equal = np.all(x == y)
            else:
                is_equal = (x == y)
            if not is_equal:
                raise AssertionError("Rank 0's and %d's result differ: '%r' vs '%r'" %
                                     (i, x, y))

#
# @mpitest decorator
#

def format_exc_info():
    import traceback
    type_, value, tb = sys.exc_info()
    msg = traceback.format_exception(type_, value, tb)
    return ''.join(msg)

def first_nonzero(arr):
    """
    Find index of first nonzero element in the 1D array `arr`, or raise
    IndexError if no such element exists.
    """
    hits = np.nonzero(arr)
    assert len(hits) == 1
    if len(hits[0]) == 0:
        raise IndexError("No non-zero elements")
    else:
        return hits[0][0]

def mpitest(nprocs):
    """
    Runs a testcase using a `nprocs`-sized subset of COMM_WORLD. Also
    synchronizes results, so that a failure or error in one process
    causes all ranks to fail or error. The algorithm is:

     - If a process fails (AssertionError) or errors (any other exception),
       it propagates that

     - If a process succeeds, it reports the error of the lowest-ranking
       process that err-ed (by raising an error containing the stack trace
       as a string). If not other processes errored, the same is repeated
       with failures. Finally, the process succeeds.
    """
    def dec(func):
        @functools.wraps(func)
        def replacement_func():
            n = MPI.COMM_WORLD.Get_size()
            rank = MPI.COMM_WORLD.Get_rank()
            if n < nprocs:
                raise RuntimeError('Number of available MPI processes (%d) '
                                   'too small' % n)
            sub_comm = MPI.COMM_WORLD.Split(0 if rank < nprocs else 1, 0)
            SUCCESS, ERROR, FAILED = range(3)
            status = SUCCESS
            exc_msg = ''
            try:
                if rank < nprocs:
                    try:
                        func(sub_comm)
                    except AssertionError:
                        status = FAILED
                        exc_msg = format_exc_info()
                        raise
                    except:
                        status = ERROR
                        exc_msg = format_exc_info()
                        raise
            finally:
                # Do communication of error results in a final block, so
                # that also erring/failing processes participate

                # First, figure out status of other nodes
                statuses = MPI.COMM_WORLD.allgather(status)
                try:
                    first_non_success = first_nonzero(statuses)
                except IndexError:
                    first_non_success_status = SUCCESS
                else:
                    # First non-success gets to broadcast it's error
                    first_non_success_status, msg = MPI.COMM_WORLD.bcast(
                        (status, exc_msg), root=first_non_success)
                    
                # Exit finally-block -- erring/failing processes return here

            # Did not return -- so raise some other process' error or failure
            fmt = '%s in MPI rank %d:\n\n"""\n%s"""\n'
            if first_non_success_status == ERROR:
                msg = fmt % ('ERROR', first_non_success, msg)
                raise RuntimeError(msg)
            elif first_non_success_status == FAILED:
                msg = fmt % ('FAILURE', first_non_success, msg)
                raise AssertionError(msg)

        return replacement_func
    return dec


