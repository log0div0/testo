from common import *

def test_sheduler_dfs_basic():
	must_succeed("testo run scheduler/dfs_basic.testo --invalidate \\*", """TESTS TO RUN:
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

def test_sheduler_depens_on_basic():
	must_succeed("testo run scheduler/depends_on_basic.testo --invalidate \\*", """TESTS TO RUN:
B
B1
B11
C
C1
A
A2
A1""")

def test_sheduler_depens_on_itself():
	must_fail("testo run scheduler/depends_on_itself.testo --invalidate \\*", """Test 'A' can't depend on itself""")

def test_sheduler_depens_on_child():
	must_fail("testo run scheduler/depends_on_child.testo --invalidate \\*", """Test 'A' can't depend on its child 'A1'""")

def test_sheduler_depens_on_cyclic():
	must_fail("testo run scheduler/depends_on_cyclic.testo --invalidate \\*", """Can't decide which test to execute first because they depens on each other: A, B, C""")

def test_sheduler_depens_on_unknown_test():
	must_fail("testo run scheduler/depends_on_unknown_test.testo --invalidate \\*", """Test A depends on unknown test B""")

def test_sheduler_depens_on_multiple_deps():
	must_succeed("testo run scheduler/depends_on_multiple_deps.testo --invalidate \\*", """TESTS TO RUN:
B
A
C""")

def test_sheduler_depens_on_unselected_test():
	must_succeed("testo run scheduler/depends_on_unselected_test.testo --test_spec C --invalidate \\*", """TESTS TO RUN:
A
B
C""")

def test_sheduler_depens_on_skip_test_on_dep_fail():
	out, err = must_fail("testo run scheduler/depends_on_skip_test_on_dep_fail.testo --invalidate \\*")
	assert "Skipping test B because his dependency A_fail is failed or skipped" in out
	assert "Skipping test C because his dependency B is failed or skipped" in out
	assert """UP-TO-DATE: 0
RUN SUCCESSFULLY: 1
FAILED: 1
	 - A_fail
SKIPPED: 2
	 - B
	 - C""" in out
