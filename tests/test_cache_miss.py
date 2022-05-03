from common import *

def test_cache_miss():
	must_succeed("testo run cache_miss/original.testo --invalidate \\*")
	must_succeed("testo run cache_miss/modified.testo", """Because of the cache loss, Testo is scheduled to run the following tests:
	- test_parent
	- test_child1
	- test_child2
	- test_child3""", input="y\n")
