
load ../common

@test "fd init snapshots" {
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "RUN SUCCESSFULLY: 3"
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "UP-TO-DATE: 3"
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/fd_config_changed.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "UP-TO-DATE: 1"
	echo $output | grep "RUN SUCCESSFULLY: 2"
	echo $output | grep -A 1 "TESTS TO RUN" | tail -n 1 | grep test_child_1
	echo $output | grep -A 2 "TESTS TO RUN" | tail -n 1 | grep test_child_2
}
