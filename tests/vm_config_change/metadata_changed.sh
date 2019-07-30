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

OUTPUT=`ISO_FILE=$BASEDIR/iso.iso $TESTO_BIN $BASEDIR/scripts/base.testo --hypervisor dummy --disable_cache_miss_prompt`

EVERYTHING_PASSED="Registering machine my_machine
TEST TO RUN
parent_test
child_test
[  0%] Preparing the environment for the test parent_test
[  0%] Creating entity my_machine
[  0%] Taking initial snapshot for entity my_machine
[  0%] Running test parent_test
[  0%] my_machine: This is parent test running
[  0%] Taking snapshot parent_test for entity my_machine
[ 50%] Test parent_test PASSED
[ 50%] Preparing the environment for the test child_test
[ 50%] Running test child_test
[ 50%] my_machine: This is child test running
[ 50%] Taking snapshot child_test for entity my_machine
[100%] Test child_test PASSED
PROCESSED TOTAL 2 TESTS IN 0h:0m:0s
UP TO DATE: 0
RUN SUCCESSFULLY: 2
FAILED: 0"

assert_output "$OUTPUT" "$EVERYTHING_PASSED"

OUTPUT=`ISO_FILE=$BASEDIR/iso.iso $TESTO_BIN $BASEDIR/scripts/base.testo --hypervisor dummy --disable_cache_miss_prompt`

EVERYTHING_UP_TO_DATE="Registering machine my_machine
[ 50%] Test parent_test is up-to-date, skipping...
[100%] Test child_test is up-to-date, skipping...
TEST TO RUN
PROCESSED TOTAL 2 TESTS IN 0h:0m:0s
UP TO DATE: 2
RUN SUCCESSFULLY: 0
FAILED: 0"

assert_output "$OUTPUT" "$EVERYTHING_UP_TO_DATE"

OUTPUT=`ISO_FILE=$BASEDIR/iso.iso $TESTO_BIN $BASEDIR/scripts/metadata_changed.testo --hypervisor dummy --disable_cache_miss_prompt`

assert_output "$OUTPUT" "$EVERYTHING_UP_TO_DATE"
