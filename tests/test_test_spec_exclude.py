from common import *

def test_simple_test_spec():
	out, err = must_succeed('testo run test_spec_exclude/test_spec_exclude.testo --test_spec test1*')
	assert "test121\n" in out
	assert "test141\n" in out
	assert "test2\n" not in out
	assert "test3\n" not in out

def test_wildcard_test_spec():
	out, err = must_succeed('testo run test_spec_exclude/test_spec_exclude.testo --test_spec *test1*')
	assert "test121\n" in out
	assert "test141\n" in out
	assert "test2\n" not in out
	assert "test22\n" not in out
	assert "test3\n" not in out

def test_multiple_test_spec():
	# IS IT OK????
	out, err = must_succeed('testo run test_spec_exclude/test_spec_exclude.testo --test_spec test1* --test_spec test2')
	assert "test121\n" not in out
	assert "test141\n" not in out
	assert "test2\n" not in out
	assert "test3\n" not in out
	assert "test22\n" not in out

def test_simple_exclude():
	out, err = must_succeed('testo run test_spec_exclude/test_spec_exclude.testo --exclude test2')
	assert "test121\n" in out
	assert "test141\n" in out
	assert "test22\n" in out
	assert "test3\n" in out
	assert "test2\n" not in out

def test_multiple_exclude():
	out, err = must_succeed('testo run test_spec_exclude/test_spec_exclude.testo --exclude test2 --exclude test1*')
	assert "test121\n" not in out
	assert "test141\n" not in out
	assert "test22\n" in out
	assert "test3\n" in out
	assert "test2\n" not in out

def test_test_spec_exclude_pipeline1():
	out, err = must_succeed('testo run test_spec_exclude/test_spec_exclude.testo --test_spec test*2* --exclude test2*')
	assert "test121\n" in out
	assert "test141\n" not in out
	assert "test22\n" not in out
	assert "test3\n" not in out
	assert "test2\n" not in out

def test_test_spec_exclude_pipeline2():
	out, err = must_succeed('testo run test_spec_exclude/test_spec_exclude.testo --exclude test2* --test_spec test*2*')
	assert "test121\n" in out
	assert "test141\n" not in out
	assert "test22\n" not in out
	assert "test3\n" not in out
	assert "test2\n" not in out
