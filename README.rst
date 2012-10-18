mpinoseutils -- Utilities for using nose and mpi4py together
============================================================

"Installation"
--------------

What you need is included in the module ``mpinoseutils.py``.

An ``__init__.py`` is provided in case you want to use ``git subtree``
to make this a sub-package of your own projects. Distribution with
your own projects is encouraged at this time, perhaps once the project
is mature it can become a dependency instead.

Usage
-----

Write tests like this::

    from mpinoseutils import *
    
    @mpitest(4) # argument is number of ranks needed for test
    def test_demo(comm):
        assert comm.Get_size() == 4
        
        # meq_: Different expected result for each rank
        meq_(comm, [0, 1, 2, 3], comm.Get_rank()) 

        allranks = comm.allgather(comm.Get_rank())
        # mprint: collective non-garbled printing
        mprint(comm, 'allranks == ', allranks)
        assert_eq_across_ranks(comm, allranks)

Running in non-collective mode
------------------------------

Then run the tests like regular. When the test is run, the decorator
will spawn MPI subprocesses using ``mpiexec`` and communicate with
rank 0 using ZeroMQ in order to execute the function and
report the results. For each ``@mpitest``, a new, isolated set of
MPI processes are spawned.

Test fixtures (``setup()``, ``teardown()`` etc.) are not run; the
function is simply executed as the only thing in the spawned Python
process.


Running in collective mode
--------------------------

Alternatively, by using ``setup.cfg`` and ``runtests.py``,
you can run all the MPI processes through all the tests::

    mpiexec -np 10 python runtests.py [args-to-nose]

The number of ranks should be the **maximum** of the ranks needed in
individual tests (i.e., the ``@mpitest`` decorator creates a
sub-communicator with the right number of ranks).

Using ``runtests.py`` rather than ``nosetests`` will silence nose
printing from every rank but the 0-rank. After each
``@mpitest``-decorated test has run, the results are gathered
to rank 0, which will raise errors on behalf of the other ranks.

License
-------

BSD 3-clause

