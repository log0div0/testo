
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

OUTPUT=`$TESTO_BIN $BASEDIR/scripts/exit_code_0.testo --hypervisor dummy --disable_cache_miss_prompt`

RETURN_CODE=$?

EVERYTHING_PASSED="Registering machine my_machine
TEST TO RUN
parent_test
child_test
another_child_test
[  0%] Preparing the environment for the test parent_test
[  0%] Creating entity my_machine
[  0%] Taking initial snapshot for entity my_machine
[  0%] Running test parent_test
[  0%] my_machine: Some string
[  0%] Taking snapshot parent_test for entity my_machine
[ 33%] Test parent_test PASSED
[ 33%] Preparing the environment for the test child_test
[ 33%] Running test child_test
[ 33%] my_machine: Some other string
[ 33%] Taking snapshot child_test for entity my_machine
[ 67%] Test child_test PASSED
[ 67%] Preparing the environment for the test another_child_test
[ 67%] Restoring snapshot parent_test for entity my_machine
[ 67%] Running test another_child_test
[ 67%] my_machine: Some other string
[ 67%] Taking snapshot another_child_test for entity my_machine
[100%] Test another_child_test PASSED
PROCESSED TOTAL 3 TESTS IN 0h:0m:0s
UP TO DATE: 0
RUN SUCCESSFULLY: 3
FAILED: 0"

assert_output "$OUTPUT" "$EVERYTHING_PASSED"
assert_output "$RETURN_CODE" "0"
