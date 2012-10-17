mpinoseutils -- Utilities for using nose and mpi4py together
============================================================

"Installation"
--------------

Copy the files ``mpinoseutils.py`` (a Python module) and
``runtests.py`` (a test-runner script) to appropriate places in your
project. Also, add this to your ``setup.cfg``::

    [nosetests]
    
    with-mpi=1


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

Then run the tests like this::

    mpiexec -np 10 python runtests.py [args-to-nose]

The number of ranks should be the **maximum** of the ranks needed
in individual tests.

Using ``runtests.py`` rather than ``nosetests`` will silence nose
printing from every rank but the 0-rank. After each
``@mpitest``-decorated test has run, the results are gathered
to rank 0, which will raise errors on behalf of the other ranks.
    

License
-------

BSD 3-clause

