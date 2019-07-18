
rm -rf ./dummy_hypervisor_files
rm -rf ./flash_drives_metadata
echo "Running test simple test"


rm -rf ./vm_metadata
mkdir ./dummy_hypervisor_files

BASEDIR=$(dirname "$0")

OUTPUT=`./out/sbin/testo $BASEDIR/simple_test.testo --hypervisor dummy`

EXPECTED_OUTPUT="Registering machine my_machine
TEST TO RUN
simple_test
[  0%] Preparing the environment for the test simple_test
[  0%] Creating machine my_machine
[  0%] Running test simple_test
[  0%] Starting virtual machine my_machine
[  0%] Taking snapshot simple_test for entity my_machine
[100%] Test simple_test PASSED
PROCESSED TOTAL 1 TESTS IN 0h:0m:0s
UP TO DATE: 0
RUN SUCCESSFULLY: 1
FAILED: 0"

if [ "$OUTPUT" != "$EXPECTED_OUTPUT" ]; then
	echo "test simle test failed"
	diff <(echo "$OUTPUT") <(echo "$EXPECTED_OUTPUT")
	exit 1
fi

OUTPUT=`./out/sbin/testo $BASEDIR/simple_test.testo --hypervisor dummy`

EXPECTED_OUTPUT="Registering machine my_machine
[100%] Test simple_test is up-to-date, skipping...
TEST TO RUN
PROCESSED TOTAL 1 TESTS IN 0h:0m:0s
UP TO DATE: 1
RUN SUCCESSFULLY: 0
FAILED: 0"

if [ "$OUTPUT" != "$EXPECTED_OUTPUT" ]; then
	echo "test simle test failed"
	diff <(echo "$OUTPUT") <(echo "$EXPECTED_OUTPUT")
	exit 1
fi

echo "Test simple test passed"
