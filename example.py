from mpinoseutils import *

@mpitest(4)
def test_demo(comm):
    assert comm.Get_size() == 4
    allranks = comm.allgather(comm.Get_rank())
    mprint(comm, 'allranks == ', allranks)
    assert_eq_across_ranks(comm, allranks)
