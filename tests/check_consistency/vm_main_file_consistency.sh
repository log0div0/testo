
BASEDIR=$(dirname "$0")

TESTO_BIN=$SBIN_DIR/testo

assert_output() {
	if [ "$1" != "$2" ]; then
		diff <(echo "$1") <(echo "$2")
		exit 1
	fi
}

rm -rf ./dummy_hypervisor_files
rm -rf ./flash_drives_metadata
rm -rf ./vm_metadata

mkdir ./dummy_hypervisor_files

OUTPUT=`$TESTO_BIN $BASEDIR/scripts/controller_consistency.testo --hypervisor dummy --cache_miss_policy accept`

EVERYTHING_PASSED="Registering machine my_machine
Registering flash my_flash
TEST TO RUN
my_test
[  0%] Preparing the environment for the test my_test
[  0%] Creating entity my_machine
[  0%] Taking initial snapshot for entity my_machine
[  0%] Creating entity my_flash
[  0%] Taking initial snapshot for entity my_flash
[  0%] Running test my_test
[  0%] Plugging flash drive my_flash in virtual machine my_machine
[  0%] Unlugging flash drive my_flash from virtual machine my_machine
[  0%] Taking snapshot my_test for entity my_machine
[  0%] Taking snapshot my_test for entity my_flash
[100%] Test my_test PASSED
PROCESSED TOTAL 1 TESTS IN 0h:0m:0s
UP TO DATE: 0
RUN SUCCESSFULLY: 1
FAILED: 0"

assert_output "$OUTPUT" "$EVERYTHING_PASSED"

OUTPUT=`$TESTO_BIN $BASEDIR/scripts/controller_consistency.testo --hypervisor dummy --cache_miss_policy accept`

EVERYTHING_IS_UP_TO_DATE="Registering machine my_machine
Registering flash my_flash
[100%] Test my_test is up-to-date, skipping...
TEST TO RUN
PROCESSED TOTAL 1 TESTS IN 0h:0m:0s
UP TO DATE: 1
RUN SUCCESSFULLY: 0
FAILED: 0"

assert_output "$OUTPUT" "$EVERYTHING_IS_UP_TO_DATE"

#delete the metadata file

rm ./vm_metadata/my_machine/my_machine

OUTPUT=`$TESTO_BIN $BASEDIR/scripts/controller_consistency.testo --hypervisor dummy --cache_miss_policy accept`

VM_CREATED_AGAIN="Registering machine my_machine
Registering flash my_flash
TEST TO RUN
my_test
[  0%] Preparing the environment for the test my_test
[  0%] Creating entity my_machine
[  0%] Taking initial snapshot for entity my_machine
[  0%] Restoring initial snapshot for entity my_flash
[  0%] Running test my_test
[  0%] Plugging flash drive my_flash in virtual machine my_machine
[  0%] Unlugging flash drive my_flash from virtual machine my_machine
[  0%] Taking snapshot my_test for entity my_machine
[  0%] Taking snapshot my_test for entity my_flash
[100%] Test my_test PASSED
PROCESSED TOTAL 1 TESTS IN 0h:0m:0s
UP TO DATE: 0
RUN SUCCESSFULLY: 1
FAILED: 0"

assert_output "$OUTPUT" "$VM_CREATED_AGAIN"

OUTPUT=`$TESTO_BIN $BASEDIR/scripts/controller_consistency.testo --hypervisor dummy --cache_miss_policy accept`

assert_output "$OUTPUT" "$EVERYTHING_IS_UP_TO_DATE"

#delete the vm itself

rm ./dummy_hypervisor_files/my_machine

OUTPUT=`$TESTO_BIN $BASEDIR/scripts/controller_consistency.testo --hypervisor dummy --cache_miss_policy accept`

assert_output "$OUTPUT" "$VM_CREATED_AGAIN"

OUTPUT=`$TESTO_BIN $BASEDIR/scripts/controller_consistency.testo --hypervisor dummy --cache_miss_policy accept`

assert_output "$OUTPUT" "$EVERYTHING_IS_UP_TO_DATE"
