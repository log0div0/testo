
load ../common

export ISO_FILE=$BATS_TEST_DIRNAME/iso.iso

@test "vm metadata order changed" {
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "RUN SUCCESSFULLY: 2"
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "UP-TO-DATE: 2"
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/metadata_changed.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "UP-TO-DATE: 2"
}

@test "iso changed" {
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "RUN SUCCESSFULLY: 2"
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "UP-TO-DATE: 2"
	echo "1" >> $BATS_TEST_DIRNAME/iso.iso
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "RUN SUCCESSFULLY: 2"
}

@test "nics changed" {
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "RUN SUCCESSFULLY: 2"
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "UP-TO-DATE: 2"
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/nics_changed.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "RUN SUCCESSFULLY: 2"
}

@test "same iso different path" {
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "RUN SUCCESSFULLY: 2"
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "UP-TO-DATE: 2"
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/iso_as_string.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "UP-TO-DATE: 2"
}

@test "network changed" {
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "RUN SUCCESSFULLY: 2"
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "UP-TO-DATE: 2"
	run $TESTO_BIN $BATS_TEST_DIRNAME/scripts/network_changed.testo --hypervisor dummy --cache_miss_policy accept
	echo $output | grep "RUN SUCCESSFULLY: 2"
}
