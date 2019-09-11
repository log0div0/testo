
load ../common

@test "vm init snapshots" {
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "RUN SUCCESSFULLY: 2"
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "UP-TO-DATE: 2"
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/vm_config_changed.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "RUN SUCCESSFULLY: 2"
}
