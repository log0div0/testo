from common import *

def test_skip_failed():
	out, err = must_fail("testo run skip_failed/skip_failed.testo --invalidate \\*")
	assert "Test test_parent PASSED" in out
	assert "Test test_child1 FAILED" in out
	assert "Skipping test test_child2 because his parent test_child1 is failed or skipped" in out
	assert "Test test_parent2 PASSED" in out

def test_skip_failed2():
	out, err = must_fail("testo run skip_failed/skip_failed2.testo --invalidate \\*")
	assert "Test test_parent FAILED" in out
	assert "Skipping test test_child1 because his parent test_parent is failed or skipped" in out
	assert "Skipping test test_child2 because his parent test_parent is failed or skipped" in out
