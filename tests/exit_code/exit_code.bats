
load ../common

@test "exit code 0" {
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/exit_code_0.testo --hypervisor dummy --cache_miss_policy accept
	[ "$status" -eq 0 ]
	echo $output | grep "FAILED: 0"
}

@test "exit code 1" {
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/exit_code_1.testo --hypervisor dummy --cache_miss_policy accept
	[ "$status" -eq 1 ]
	echo $output | grep "FAILED: 1"
	echo $output | grep -A1 "FAILED: 1" | tail -n1 | grep "child_test"
}
