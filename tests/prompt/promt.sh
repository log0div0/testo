
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

OUTPUT=`$TESTO_BIN $BASEDIR/scripts/base.testo --hypervisor dummy`

EVERYTHING_PASSED="Registering machine my_machine1
Registering machine my_machine2
Registering machine my_machine3
Registering machine my_machine4
Registering flash my_flash
TEST TO RUN
branch1_parent1
branch2_parent
branch1_parent2
branch1_child1
branch1_child2
branch2_child1
branch2_child2
[  0%] Preparing the environment for the test branch1_parent1
[  0%] Creating entity my_machine1
[  0%] Taking initial snapshot for entity my_machine1
[  0%] Running test branch1_parent1
[  0%] my_machine1: branch1_parent1
[  0%] Taking snapshot branch1_parent1 for entity my_machine1
[ 14%] Test branch1_parent1 PASSED
[ 14%] Preparing the environment for the test branch2_parent
[ 14%] Creating entity my_machine3
[ 14%] Taking initial snapshot for entity my_machine3
[ 14%] Running test branch2_parent
[ 14%] my_machine3: branch1_parent1
[ 14%] Taking snapshot branch2_parent for entity my_machine3
[ 29%] Test branch2_parent PASSED
[ 29%] Preparing the environment for the test branch1_parent2
[ 29%] Creating entity my_machine2
[ 29%] Taking initial snapshot for entity my_machine2
[ 29%] Running test branch1_parent2
[ 29%] my_machine2: branch1_parent1
[ 29%] Taking snapshot branch1_parent2 for entity my_machine2
[ 43%] Test branch1_parent2 PASSED
[ 43%] Preparing the environment for the test branch1_child1
[ 43%] Running test branch1_child1
[ 43%] my_machine1: branch1_child1
[ 43%] Taking snapshot branch1_child1 for entity my_machine1
[ 43%] Taking snapshot branch1_child1 for entity my_machine2
[ 57%] Test branch1_child1 PASSED
[ 57%] Preparing the environment for the test branch1_child2
[ 57%] Restoring snapshot branch1_parent1 for entity my_machine1
[ 57%] Restoring snapshot branch1_parent2 for entity my_machine2
[ 57%] Creating entity my_flash
[ 57%] Taking initial snapshot for entity my_flash
[ 57%] Running test branch1_child2
[ 57%] my_machine2: branch1_child2
[ 57%] Plugging flash drive my_flash in virtual machine my_machine2
[ 57%] Unlugging flash drive my_flash from virtual machine my_machine2
[ 57%] Taking snapshot branch1_child2 for entity my_machine1
[ 57%] Taking snapshot branch1_child2 for entity my_machine2
[ 57%] Taking snapshot branch1_child2 for entity my_flash
[ 71%] Test branch1_child2 PASSED
[ 71%] Preparing the environment for the test branch2_child1
[ 71%] Creating entity my_machine4
[ 71%] Taking initial snapshot for entity my_machine4
[ 71%] Running test branch2_child1
[ 71%] my_machine4: branch2_child1
[ 71%] Taking snapshot branch2_child1 for entity my_machine3
[ 71%] Taking snapshot branch2_child1 for entity my_machine4
[ 86%] Test branch2_child1 PASSED
[ 86%] Preparing the environment for the test branch2_child2
[ 86%] Running test branch2_child2
[ 86%] my_machine3: branch2_child2
[ 86%] Taking snapshot branch2_child2 for entity my_machine3
[ 86%] Taking snapshot branch2_child2 for entity my_machine4
[100%] Test branch2_child2 PASSED
PROCESSED TOTAL 7 TESTS IN 0h:0m:0s
UP TO DATE: 0
RUN SUCCESSFULLY: 7
FAILED: 0"

assert_output "$OUTPUT" "$EVERYTHING_PASSED"

#ok, now we need to invalidate some cache
#1) Let's invalidate vm and ignore some tests

# OUTPUT=`$TESTO_BIN $BASEDIR/scripts/vm_config_changed.testo --hypervisor dummy --cache_miss_policy skip_branch`

# BRANCH2_INVALIDATED="Registering machine my_machine1
# Registering machine my_machine2
# Registering machine my_machine3
# Registering machine my_machine4
# Registering flash my_flash
# [ 20%] Test branch1_parent1 is up-to-date, skipping...
# [ 40%] Test branch2_parent is up-to-date, skipping...
# [ 60%] Test branch1_parent2 is up-to-date, skipping...
# [ 80%] Test branch1_child1 is up-to-date, skipping...
# [100%] Test branch1_child2 is up-to-date, skipping...
# TEST TO RUN
# PROCESSED TOTAL 7 TESTS IN 0h:0m:0s
# UP TO DATE: 5
# LOST CACHE, BUT SKIPPED: 2
# 	 -branch2_child1
# 	 -branch2_child2
# RUN SUCCESSFULLY: 0
# FAILED: 0"

# assert_output "$OUTPUT" "$BRANCH2_INVALIDATED"

# #2) Now let's invalidate flash drive

# OUTPUT=`$TESTO_BIN $BASEDIR/scripts/fd_config_changed.testo --hypervisor dummy --cache_miss_policy skip_branch`

# BRANCH1_CHILD2_INVALIDATED="Registering machine my_machine1
# Registering machine my_machine2
# Registering machine my_machine3
# Registering machine my_machine4
# Registering flash my_flash
# [ 17%] Test branch1_parent1 is up-to-date, skipping...
# [ 33%] Test branch2_parent is up-to-date, skipping...
# [ 50%] Test branch1_parent2 is up-to-date, skipping...
# [ 67%] Test branch1_child1 is up-to-date, skipping...
# [ 83%] Test branch2_child1 is up-to-date, skipping...
# [100%] Test branch2_child2 is up-to-date, skipping...
# TEST TO RUN
# PROCESSED TOTAL 7 TESTS IN 0h:0m:0s
# UP TO DATE: 6
# LOST CACHE, BUT SKIPPED: 1
# 	 -branch1_child2
# RUN SUCCESSFULLY: 0
# FAILED: 0"

# assert_output "$OUTPUT" "$BRANCH1_CHILD2_INVALIDATED"
