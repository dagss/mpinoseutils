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

def _spawn_mpi_test(nprocs, func):
    import subprocess
    import sys
    import os
    import zmq

    # Since the output terminals are used for lots of debug output etc., we use
    # ZeroMQ to communicate with the workers.
    zctx = zmq.Context()
    socket = zctx.socket(zmq.REQ)
    port = socket.bind_to_random_port("tcp://*")
    cmd = 'import %s as mod; mod._mpi_worker("tcp://*:%d")' % (__name__, port)
    env = dict(os.environ)
    env['PYTHONPATH'] = ':'.join(sys.path)
    child = subprocess.Popen(['mpiexec', '-np', str(nprocs), sys.executable,
                              '-c', cmd], env=env)

    # Call on the root worker; it will use MPI to scatter func and gather result
    socket.send_pyobj((func.__module__, func.__name__))
    result = socket.recv_pyobj()
    socket.send_pyobj('stop')
    socket.recv_pyobj()
    # TODO: If nose is capturing, gather output from child and forward to nose
    child.wait()
    _raise_condition(*result)

def _mpi_worker(addr):
    import importlib
    import zmq
    from cPickle import loads

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        zctx = zmq.Context()
        socket = zctx.socket(zmq.REP)
        socket.connect(addr)
        pickled_func_info = socket.recv()
    else:
        pickled_func_info = None
    pickled_func_info = MPI.COMM_WORLD.bcast(pickled_func_info, root=0)
    module_name, func_name = loads(pickled_func_info)
    mod = importlib.import_module(module_name)
    func = getattr(mod, func_name)    
    status = func(_return_status=True)
    if rank == 0:
        socket.send_pyobj(status)
        # Wait for termination message
        assert socket.recv_pyobj() == 'stop'
        socket.send_pyobj('')
    # All processes wait until they can terminate
    MPI.COMM_WORLD.barrier()
        
def _raise_condition(first_non_success_status, failing_rank, msg):
    fmt = '%s in MPI rank %d:\n\n"""\n%s"""\n'
    if first_non_success_status == 'ERROR':
        msg = fmt % ('ERROR', failing_rank, msg)
        raise RuntimeError(msg)
    elif first_non_success_status == 'FAILED':
        msg = fmt % ('FAILURE', failing_rank, msg)
        raise AssertionError(msg)
    elif first_non_success_status == 'SUCCESS':
        pass
    else:
        assert False

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
        def replacement_func(_return_status=False):
            from cPickle import dumps
            
            n = MPI.COMM_WORLD.Get_size()
            rank = MPI.COMM_WORLD.Get_rank()
            import os
            if n == 1:
                # Not in collective mode; spawn a sub-process to run this test
                return _spawn_mpi_test(nprocs, func)
            
            if n < nprocs:
                raise RuntimeError('Number of available MPI processes (%d) '
                                   'too small' % n)
            sub_comm = MPI.COMM_WORLD.Split(0 if rank < nprocs else 1, 0)
            status = 'SUCCESS'
            exc_msg = ''

            try:
                if rank < nprocs:
                    try:
                        func(sub_comm)
                    except AssertionError:
                        status = 'FAILED'
                        exc_msg = format_exc_info()
                        if not _return_status:
                            raise
                    except:
                        status = 'ERROR'
                        exc_msg = format_exc_info()
                        if not _return_status:
                            raise
            finally:
                # Do communication of error results in a final block, so
                # that also erring/failing processes participate

                # First, figure out status of other nodes
                statuses = MPI.COMM_WORLD.allgather(status)
                try:
                    first_failing_rank = first_nonzero(statuses)
                except IndexError:
                    first_failing_rank = -1
                    first_non_success_status = 'SUCCESS'
                else:
                    # First non-success gets to broadcast it's error
                    first_non_success_status, msg = MPI.COMM_WORLD.bcast(
                        (status, exc_msg), root=first_failing_rank)

            if _return_status:
                return (first_non_success_status, first_failing_rank, msg)
            else:
                _raise_condition(first_non_success_status, first_failing_rank, msg)

        return replacement_func
    return dec


