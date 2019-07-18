BASEDIR=$(dirname "$0")

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

OUTPUT=`./out/sbin/testo $BASEDIR/scripts/base.testo --hypervisor dummy`

EVERYTHING_PASSED="Registering machine my_machine
TEST TO RUN
parent_test
child_test
[  0%] Preparing the environment for the test parent_test
[  0%] Creating machine my_machine
[  0%] Running test parent_test
[  0%] Starting virtual machine my_machine
[  0%] Stopping virtual machine my_machine
[  0%] Resolving var host_name
[  0%] my_machine: my-machine
[  0%] my_machine: Some string
[  0%] Taking snapshot parent_test for entity my_machine
[ 50%] Test parent_test PASSED
[ 50%] Preparing the environment for the test child_test
[ 50%] Running test child_test
[ 50%] Starting virtual machine my_machine
[ 50%] Resolving var login
[ 50%] my_machine: my_machine_login
[ 50%] Taking snapshot child_test for entity my_machine
[100%] Test child_test PASSED
PROCESSED TOTAL 2 TESTS IN 0h:0m:0s
UP TO DATE: 0
RUN SUCCESSFULLY: 2
FAILED: 0"

assert_output "$OUTPUT" "$EVERYTHING_PASSED"

OUTPUT=`./out/sbin/testo $BASEDIR/scripts/base.testo --hypervisor dummy`

EVERYTHING_UP_TO_DATE="Registering machine my_machine
[ 50%] Test parent_test is up-to-date, skipping...
[100%] Test child_test is up-to-date, skipping...
TEST TO RUN
PROCESSED TOTAL 2 TESTS IN 0h:0m:0s
UP TO DATE: 2
RUN SUCCESSFULLY: 0
FAILED: 0"

assert_output "$OUTPUT" "$EVERYTHING_UP_TO_DATE"

OUTPUT=`./out/sbin/testo $BASEDIR/scripts/no_snapshots.testo --hypervisor dummy`

RUN_ONLY_CHILD="Registering machine my_machine
[ 50%] Test parent_test is up-to-date, skipping...
TEST TO RUN
child_test
[ 50%] Preparing the environment for the test child_test
[ 50%] Restoring snapshot parent_test for entity my_machine
[ 50%] Running test child_test
[ 50%] Starting virtual machine my_machine
[ 50%] Resolving var login
[ 50%] my_machine: my_machine_login
[ 50%] Taking snapshot child_test for entity my_machine
[100%] Test child_test PASSED
PROCESSED TOTAL 2 TESTS IN 0h:0m:0s
UP TO DATE: 1
RUN SUCCESSFULLY: 1
FAILED: 0"

assert_output "$OUTPUT" "$RUN_ONLY_CHILD"
