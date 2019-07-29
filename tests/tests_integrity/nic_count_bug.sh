
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

OUTPUT=`$TESTO_BIN $BASEDIR/scripts/nic_count_bug.testo --hypervisor dummy`

EVERYTHING_PASSED="Registering machine my_machine
TEST TO RUN
parent_test
[  0%] Preparing the environment for the test parent_test
[  0%] Creating entity my_machine
[  0%] Taking initial snapshot for entity my_machine
[  0%] Running test parent_test
[  0%] Resolving var vm_nic_count
[  0%] my_machine: Is zero
[  0%] Taking snapshot parent_test for entity my_machine
[100%] Test parent_test PASSED
PROCESSED TOTAL 1 TESTS IN 0h:0m:0s
UP TO DATE: 0
RUN SUCCESSFULLY: 1
FAILED: 0"

assert_output "$OUTPUT" "$EVERYTHING_PASSED"

OUTPUT=`$TESTO_BIN $BASEDIR/scripts/nic_count_bug.testo --hypervisor dummy`

EVERYTHING_UP_TO_DATE="Registering machine my_machine
[100%] Test parent_test is up-to-date, skipping...
TEST TO RUN
PROCESSED TOTAL 1 TESTS IN 0h:0m:0s
UP TO DATE: 1
RUN SUCCESSFULLY: 0
FAILED: 0"

assert_output "$OUTPUT" "$EVERYTHING_UP_TO_DATE"
