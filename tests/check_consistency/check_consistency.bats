
load ../common

@test "check fd consistency" {
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/controller_consistency.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "Creating entity my_flash"
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/controller_consistency.testo --hypervisor dummy --cache_miss_policy accept
	(! echo $output | grep "Creating entity my_flash")
	rm ./flash_drives_metadata/my_flash/my_flash
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/controller_consistency.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "Creating entity my_flash"
	rm ./dummy_hypervisor_files/my_flash
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/controller_consistency.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "Creating entity my_flash"
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/controller_consistency.testo --hypervisor dummy --cache_miss_policy accept
	(! echo $output | grep "Creating entity my_flash")
}

@test "check vm consistency" {
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/controller_consistency.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "Creating entity my_machine"
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/controller_consistency.testo --hypervisor dummy --cache_miss_policy accept
	(! echo $output | grep "Creating entity my_machine")
	rm ./vm_metadata/my_machine/my_machine
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/controller_consistency.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "Creating entity my_machine"
	rm ./dummy_hypervisor_files/my_machine
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/controller_consistency.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "Creating entity my_machine"
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/controller_consistency.testo --hypervisor dummy --cache_miss_policy accept
	(! echo $output | grep "Creating entity my_machine")
}
