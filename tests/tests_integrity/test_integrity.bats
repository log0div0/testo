
load ../common

@test "vm metadata value changed" {
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "RUN SUCCESSFULLY: 2"
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "UP-TO-DATE: 2"
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/metadata_changed.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "UP-TO-DATE: 1"
	echo $output | grep "RUN SUCCESSFULLY: 1"
	echo $output | grep -A 1 "TESTS TO RUN" | tail -n 1 | grep child_test
}

@test "nic count bug" {
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/nic_count_bug.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "RUN SUCCESSFULLY: 1"
	echo $output | grep "my_machine: Is zero"
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/nic_count_bug.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "UP-TO-DATE: 1"
}

@test "no snapshots" {
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "RUN SUCCESSFULLY: 2"
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "UP-TO-DATE: 2"
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/no_snapshots.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "RUN SUCCESSFULLY: 1"
	echo $output | grep "UP-TO-DATE: 1"
	echo $output | grep -A 1 "TESTS TO RUN" | tail -n 1 | grep child_test
}
