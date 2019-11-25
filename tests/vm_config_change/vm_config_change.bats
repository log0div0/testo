
load ../common

export ISO_FILE=$BATS_TEST_DIRNAME/iso.iso

@test "param_changed" {
	cp $BATS_TEST_DIRNAME/iso_original.iso $BATS_TEST_DIRNAME/iso.iso
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept --param ISO_FILE $BATS_TEST_DIRNAME/iso.iso
	echo $output | grep "RUN SUCCESSFULLY: 2"
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept --param ISO_FILE $BATS_TEST_DIRNAME/iso.iso
	echo $output | grep "UP-TO-DATE: 2"
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/param_changed.testo --hypervisor dummy --cache_miss_policy accept --param ISO_FILE $BATS_TEST_DIRNAME/iso.iso
	echo $output | grep "RUN SUCCESSFULLY: 2"
	rm -f $BATS_TEST_DIRNAME/iso.iso
}

@test "same_param_from_commandline" {
	cp $BATS_TEST_DIRNAME/iso_original.iso $BATS_TEST_DIRNAME/iso.iso
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept --param ISO_FILE $BATS_TEST_DIRNAME/iso.iso
	echo $output | grep "RUN SUCCESSFULLY: 2"
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept --param ISO_FILE $BATS_TEST_DIRNAME/iso.iso
	echo $output | grep "UP-TO-DATE: 2"
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/same_param_from_commandline.testo --hypervisor dummy --cache_miss_policy accept --param ISO_FILE $BATS_TEST_DIRNAME/iso.iso --param mac 52:54:00:00:00:11
	echo $output | grep "UP-TO-DATE: 2"
	rm -f $BATS_TEST_DIRNAME/iso.iso
}

@test "iso changed" {
	cp $BATS_TEST_DIRNAME/iso_original.iso $BATS_TEST_DIRNAME/iso.iso
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept --param ISO_FILE $BATS_TEST_DIRNAME/iso.iso
	echo $output | grep "RUN SUCCESSFULLY: 2"
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept --param ISO_FILE $BATS_TEST_DIRNAME/iso.iso
	echo $output | grep "UP-TO-DATE: 2"
	echo "1" >> $BATS_TEST_DIRNAME/iso.iso
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept --param ISO_FILE $BATS_TEST_DIRNAME/iso.iso
	echo $output | grep "RUN SUCCESSFULLY: 2"
	rm -f $BATS_TEST_DIRNAME/iso.iso
}

@test "nics changed" {
	cp $BATS_TEST_DIRNAME/iso_original.iso $BATS_TEST_DIRNAME/iso.iso
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept --param ISO_FILE $BATS_TEST_DIRNAME/iso.iso
	echo $output | grep "RUN SUCCESSFULLY: 2"
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept --param ISO_FILE $BATS_TEST_DIRNAME/iso.iso
	echo $output | grep "UP-TO-DATE: 2"
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/nics_changed.testo --hypervisor dummy --cache_miss_policy accept --param ISO_FILE $BATS_TEST_DIRNAME/iso.iso
	echo $output | grep "RUN SUCCESSFULLY: 2"
	rm -f $BATS_TEST_DIRNAME/iso.iso
}

@test "same iso different path" {
	cp $BATS_TEST_DIRNAME/iso_original.iso $BATS_TEST_DIRNAME/iso.iso
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept --param ISO_FILE $BATS_TEST_DIRNAME/iso.iso
	echo $output | grep "RUN SUCCESSFULLY: 2"
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept --param ISO_FILE $BATS_TEST_DIRNAME/iso.iso
	echo $output | grep "UP-TO-DATE: 2"
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/iso_as_string.testo --hypervisor dummy --cache_miss_policy accept --param ISO_FILE $BATS_TEST_DIRNAME/iso.iso
	echo $output | grep "UP-TO-DATE: 2"
	rm -f $BATS_TEST_DIRNAME/iso.iso
}

@test "network changed" {
	cp $BATS_TEST_DIRNAME/iso_original.iso $BATS_TEST_DIRNAME/iso.iso
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept --param ISO_FILE $BATS_TEST_DIRNAME/iso.iso
	echo $output | grep "RUN SUCCESSFULLY: 2"
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/base.testo --hypervisor dummy --cache_miss_policy accept --param ISO_FILE $BATS_TEST_DIRNAME/iso.iso
	echo $output | grep "UP-TO-DATE: 2"
	run $TESTO_BIN run $BATS_TEST_DIRNAME/scripts/network_changed.testo --hypervisor dummy --cache_miss_policy accept --param ISO_FILE $BATS_TEST_DIRNAME/iso.iso
	echo $output | grep "RUN SUCCESSFULLY: 2"
	rm -f $BATS_TEST_DIRNAME/iso.iso
}
