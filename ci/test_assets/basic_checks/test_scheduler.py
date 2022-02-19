from common import *

def test_sheduler_dfs():
	must_succeed("testo run scheduler/dfs.testo --invalidate \\*", """TESTS TO RUN:
root
A
A1
A2
B
B1
B2""")

def test_sheduler_dfs_multiple_parents():
	must_succeed("testo run scheduler/dfs_multiple_parents.testo --invalidate \\*", """TESTS TO RUN:
superroot
C
C1
root
A
A1
A2
B
B1
B2
C2
D
D1
D2""")
