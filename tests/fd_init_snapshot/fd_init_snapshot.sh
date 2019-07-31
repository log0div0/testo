
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

OUTPUT=`$TESTO_BIN $BASEDIR/scripts/base.testo --hypervisor dummy --cache_miss_policy accept`

EVERYTHING_PASSED="Registering machine my_machine
Registering flash my_flash
TEST TO RUN
test_parent
test_child_1
test_child_2
[  0%] Preparing the environment for the test test_parent
[  0%] Creating entity my_machine
[  0%] Taking initial snapshot for entity my_machine
[  0%] Running test test_parent
[  0%] my_machine: test_parent
[  0%] Taking snapshot test_parent for entity my_machine
[ 33%] Test test_parent PASSED
[ 33%] Preparing the environment for the test test_child_1
[ 33%] Creating entity my_flash
[ 33%] Taking initial snapshot for entity my_flash
[ 33%] Running test test_child_1
[ 33%] Plugging flash drive my_flash in virtual machine my_machine
[ 33%] Unlugging flash drive my_flash from virtual machine my_machine
[ 33%] Taking snapshot test_child_1 for entity my_flash
[ 33%] Taking snapshot test_child_1 for entity my_machine
[ 67%] Test test_child_1 PASSED
[ 67%] Preparing the environment for the test test_child_2
[ 67%] Restoring snapshot test_parent for entity my_machine
[ 67%] Restoring initial snapshot for entity my_flash
[ 67%] Running test test_child_2
[ 67%] Plugging flash drive my_flash in virtual machine my_machine
[ 67%] Unlugging flash drive my_flash from virtual machine my_machine
[ 67%] Taking snapshot test_child_2 for entity my_flash
[ 67%] Taking snapshot test_child_2 for entity my_machine
[100%] Test test_child_2 PASSED
PROCESSED TOTAL 3 TESTS IN 0h:0m:0s
UP TO DATE: 0
RUN SUCCESSFULLY: 3
FAILED: 0"

assert_output "$OUTPUT" "$EVERYTHING_PASSED"

OUTPUT=`$TESTO_BIN $BASEDIR/scripts/base.testo --hypervisor dummy --cache_miss_policy accept`

EVERYTHING_UP_TO_DATE="Registering machine my_machine
Registering flash my_flash
[ 33%] Test test_parent is up-to-date, skipping...
[ 67%] Test test_child_1 is up-to-date, skipping...
[100%] Test test_child_2 is up-to-date, skipping...
TEST TO RUN
PROCESSED TOTAL 3 TESTS IN 0h:0m:0s
UP TO DATE: 3
RUN SUCCESSFULLY: 0
FAILED: 0"

assert_output "$OUTPUT" "$EVERYTHING_UP_TO_DATE"

OUTPUT=`$TESTO_BIN $BASEDIR/scripts/fd_config_changed.testo --hypervisor dummy --cache_miss_policy accept`

RUN_CHILDREN="Registering machine my_machine
Registering flash my_flash
[ 33%] Test test_parent is up-to-date, skipping...
TEST TO RUN
test_child_1
test_child_2
[ 33%] Preparing the environment for the test test_child_1
[ 33%] Restoring snapshot test_parent for entity my_machine
[ 33%] Creating entity my_flash
[ 33%] Taking initial snapshot for entity my_flash
[ 33%] Running test test_child_1
[ 33%] Plugging flash drive my_flash in virtual machine my_machine
[ 33%] Unlugging flash drive my_flash from virtual machine my_machine
[ 33%] Taking snapshot test_child_1 for entity my_flash
[ 33%] Taking snapshot test_child_1 for entity my_machine
[ 67%] Test test_child_1 PASSED
[ 67%] Preparing the environment for the test test_child_2
[ 67%] Restoring snapshot test_parent for entity my_machine
[ 67%] Restoring initial snapshot for entity my_flash
[ 67%] Running test test_child_2
[ 67%] Plugging flash drive my_flash in virtual machine my_machine
[ 67%] Unlugging flash drive my_flash from virtual machine my_machine
[ 67%] Taking snapshot test_child_2 for entity my_flash
[ 67%] Taking snapshot test_child_2 for entity my_machine
[100%] Test test_child_2 PASSED
PROCESSED TOTAL 3 TESTS IN 0h:0m:0s
UP TO DATE: 1
RUN SUCCESSFULLY: 2
FAILED: 0"

assert_output "$OUTPUT" "$RUN_CHILDREN"
