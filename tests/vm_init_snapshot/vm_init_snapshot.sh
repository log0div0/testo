
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

OUTPUT=`$TESTO_BIN $BASEDIR/scripts/base.testo --hypervisor dummy --disable_cache_miss_prompt`

EVERYTHING_PASSED="Registering machine my_machine
TEST TO RUN
test1
test2
[  0%] Preparing the environment for the test test1
[  0%] Creating entity my_machine
[  0%] Taking initial snapshot for entity my_machine
[  0%] Running test test1
[  0%] my_machine: test1
[  0%] Taking snapshot test1 for entity my_machine
[ 50%] Test test1 PASSED
[ 50%] Preparing the environment for the test test2
[ 50%] Restoring initial snapshot for entity my_machine
[ 50%] Running test test2
[ 50%] my_machine: test2
[ 50%] Taking snapshot test2 for entity my_machine
[100%] Test test2 PASSED
PROCESSED TOTAL 2 TESTS IN 0h:0m:0s
UP TO DATE: 0
RUN SUCCESSFULLY: 2
FAILED: 0"

assert_output "$OUTPUT" "$EVERYTHING_PASSED"

OUTPUT=`$TESTO_BIN $BASEDIR/scripts/base.testo --hypervisor dummy --disable_cache_miss_prompt`

EVERYTHING_UP_TO_DATE="Registering machine my_machine
[ 50%] Test test1 is up-to-date, skipping...
[100%] Test test2 is up-to-date, skipping...
TEST TO RUN
PROCESSED TOTAL 2 TESTS IN 0h:0m:0s
UP TO DATE: 2
RUN SUCCESSFULLY: 0
FAILED: 0"

assert_output "$OUTPUT" "$EVERYTHING_UP_TO_DATE"

OUTPUT=`$TESTO_BIN $BASEDIR/scripts/vm_config_changed.testo --hypervisor dummy --disable_cache_miss_prompt`

EVERYTHING_PASSED="Registering machine my_machine
TEST TO RUN
test1
test2
[  0%] Preparing the environment for the test test1
[  0%] Creating entity my_machine
[  0%] Taking initial snapshot for entity my_machine
[  0%] Running test test1
[  0%] my_machine: test1
[  0%] Taking snapshot test1 for entity my_machine
[ 50%] Test test1 PASSED
[ 50%] Preparing the environment for the test test2
[ 50%] Restoring initial snapshot for entity my_machine
[ 50%] Running test test2
[ 50%] my_machine: test2
[ 50%] Taking snapshot test2 for entity my_machine
[100%] Test test2 PASSED
PROCESSED TOTAL 2 TESTS IN 0h:0m:0s
UP TO DATE: 0
RUN SUCCESSFULLY: 2
FAILED: 0"

assert_output "$OUTPUT" "$EVERYTHING_PASSED"
