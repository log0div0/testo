
load ../common

@test "prompt" {
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "RUN SUCCESSFULLY: 7"
	echo $output | grep -A 1 "TESTS TO RUN" | tail -n 1 | grep branch1_parent1
	echo $output | grep -A 2 "TESTS TO RUN" | tail -n 1 | grep branch2_parent
	echo $output | grep -A 3 "TESTS TO RUN" | tail -n 1 | grep branch1_parent2
	echo $output | grep -A 4 "TESTS TO RUN" | tail -n 1 | grep branch1_child1
	echo $output | grep -A 5 "TESTS TO RUN" | tail -n 1 | grep branch1_child2
	echo $output | grep -A 6 "TESTS TO RUN" | tail -n 1 | grep branch2_child1
	echo $output | grep -A 7 "TESTS TO RUN" | tail -n 1 | grep branch2_child2

	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/vm_config_changed.testo --hypervisor dummy --cache_miss_policy skip_branch
	echo $output | grep "UP-TO-DATE: 5"
	echo $output | grep "LOST CACHE, BUT SKIPPED: 2"
	echo $output | grep -A 1 "LOST CACHE, BUT SKIPPED: 2" | tail -n 1 | grep branch2_child1
	echo $output | grep -A 2 "LOST CACHE, BUT SKIPPED: 2" | tail -n 1 | grep branch2_child2

	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/fd_config_changed.testo --hypervisor dummy --cache_miss_policy skip_branch
	echo $output | grep "UP-TO-DATE: 6"
	echo $output | grep "LOST CACHE, BUT SKIPPED: 1"
	echo $output | grep -A 1 "LOST CACHE, BUT SKIPPED: 1" | tail -n 1 | grep branch1_child2
}
