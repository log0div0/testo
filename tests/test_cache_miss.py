from common import *
import os.path

def test_cache_miss_regular():
	must_succeed("testo run cache_miss/original.testo --invalidate \\*")
	must_succeed("testo run cache_miss/modified.testo", """Because of the cache loss, Testo is scheduled to run the following tests:
	- test_parent
	- test_child1
	- test_child2
	- test_child3""", input="y\n")

def test_cache_miss_corrupted_metadata_1():
	must_succeed("testo run cache_miss/original.testo --invalidate \\*")

	assert os.path.exists("/var/lib/libvirt/testo/vm_metadata/my_ubuntu_server/my_ubuntu_server_test_parent")
	open("/var/lib/libvirt/testo/vm_metadata/my_ubuntu_server/my_ubuntu_server_test_parent", "w").close()

	must_succeed("testo run cache_miss/modified.testo", """TESTS TO RUN:
test_parent
test_child1
test_child3
test_child2
""")

def test_cache_miss_corrupted_metadata_2():
	must_succeed("testo run cache_miss/original.testo --invalidate \\*")

	assert os.path.exists("/var/lib/libvirt/testo/vm_metadata/my_ubuntu_server/my_ubuntu_server_test_parent")
	open("/var/lib/libvirt/testo/vm_metadata/my_ubuntu_server/my_ubuntu_server_test_parent", "w").close()

	must_succeed("testo run cache_miss/modified.testo --invalidate \\*", """TESTS TO RUN:
test_parent
test_child1
test_child3
test_child2
""")

def test_cache_miss_fd_regular():
	must_succeed("testo run cache_miss/fd_original.testo --invalidate \\*")
	must_succeed("testo run cache_miss/fd_modified.testo", """Because of the cache loss, Testo is scheduled to run the following tests:
	- test_parent
	- test_child1
	- test_child2
	- test_child3""", input="y\n")

def test_cache_miss_fd_corrupted_metadata_1():
	must_succeed("testo run cache_miss/fd_original.testo --invalidate \\*")

	assert os.path.exists("/var/lib/libvirt/testo/fd_metadata/foo/foo_test_parent")
	open("/var/lib/libvirt/testo/fd_metadata/foo/foo_test_parent", "w").close()

	must_succeed("testo run cache_miss/fd_modified.testo", """TESTS TO RUN:
test_parent
test_child1
test_child3
test_child2
""")

def test_cache_miss_fd_corrupted_metadata_2():
	must_succeed("testo run cache_miss/fd_original.testo --invalidate \\*")

	assert os.path.exists("/var/lib/libvirt/testo/fd_metadata/foo/foo_test_parent")
	open("/var/lib/libvirt/testo/fd_metadata/foo/foo_test_parent", "w").close()

	must_succeed("testo run cache_miss/fd_modified.testo --invalidate \\*", """TESTS TO RUN:
test_parent
test_child1
test_child3
test_child2
""")
