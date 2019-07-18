
BASEDIR=$(dirname "$0")

for dir in $BASEDIR/*/ ; do
	for sh in $dir*.sh; do
		echo "Running test" $sh
		if /bin/bash $sh; then
			echo "Test" $sh "passed"
		else
			echo "Test" $sh "failed"
			exit 1
		fi
	done
done

echo "All tests are OK"