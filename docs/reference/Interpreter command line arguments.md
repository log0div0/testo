# Running Testo

## Interpreter arguments

The base component of Testo Framework is the `testo` interpreter.

The interpreter can be run in two modes: tests run mode (`testo run`) and entity clean mode (`testo clean`).

### Tests run mode

SYNOPSYS

```text
testo run <input file | input folder>... [--param <param_name> <param_value>]... \
  [--prefix <prefix>] [--stop_on_fail] [--assume_yes] [--test_spec <wildcard pattern>]... \
  [--exclude <wildcard pattern>]... [--invalidate <wildcard pattern>] \
  [--report_format <format>] [--report_folder </path/to/folder>] \
  [--content_cksum_maxsize <Size in Megabytes>] \
  [--nn_service <ip:port>] \
  [--ignore_repl] [--skip_tests_with_repl] \
  [--hypervisor <hypervisor type>] [--dry]
```

-   `input_file` or `input_folder`: Path to a file or a folder containing test scripts. In case a folder is specified, all the subfolders will be searched for `.testo` files as well. Several paths can be specified at once.
-   `param <param_name> <param_value>`: Define a parameter named `param_name` with the value `param_value`. The paramenter value will be visible inside test scenarios alongside with static-defined parameters.
-   `prefix <prefix>`: Add a prefix to the names of all the declared entities (virtual machines, flash drives and networks). Prefixes may be considered as a namespace alternative - with them you can create numerous independent virtual test benches that may have the same entity names. Prefixes also allow you to create several independent instances of the same virtual test bench.
-   `stop_on_fail`: Stop the test scripts run when any error occurs.
-   `assume_yes`: Disable the cache miss warning when starting test scenarios.
-   `test_spec <wildcard pattern>`: Run only the tests which names match specified pattern. Wildcard pattern format is listed below. See the tests queueing algorithm [here](#tests-queueing-algorithm).
-   `exclude <wildcard pattern>`: Don't run the tests which names match specified pattern. Wildcard pattern format is listed below. See the tests queueing algorithm [here](#tests-queueing-algorithm).
-   `invalidate <wildcard pattern>`: Force the cache invalidation for the tests which names match specified pattern. Wildcard pattern format is listed below.
-   `report_format <format>`: Supported report formats are `allure`, `native_remote` and `native_local`.
-   `report_folder </path/to/folder>`: Save the test result in the specified folder. For `native_remote` format this param should contain a tcp endpoint to connect to.
-   `content_cksum_maxsize <Size in Megabytes>`: Specify the maximum size for files to be checksummed based on their actual contents, not the last modify timestamp. See more [here](Tests.md#validating-the-test-cache). Default value: 1 Megabyte.
-   `nn_server <ip:port>`: The address of `testo-nn-server` service in `ip-addr:port` format. Default value is `127.0.0.1:8156`
-   `hypervisor <hypervisor type>`: Specify the backend hypervisor. At the moment there's only one fully supported hypervisor - `qemu`. This is an optinal argument. Default value will be determined based on the current host Operating System.
-   `ignore_repl` - Skip any `repl` action while test execution.
-   `skip_tests_with_repl` - Do not run tests that contain at least one `repl` action.
-   `dry`: Run the tests in "dry" mode. In this mode the syntax and semantic checks are performed on test scripts, but the actual tests running is never invoked. Also in this mode the cache is invalidated for the tests specified in `invalidate` command line argument.

**Return value**
-   0 - all queued tests are completed successfully.
-   1 - at lease on of the queued tests failed.
-   2 - Syntax or semantic checks fail.

> If `testo` is run with the `--stop_on_fail` command line argument, then in case of any error the value "2" is returned

### Entity clean mode

SYNOPSYS

```text
testo clean [--prefix <prefix>] [--assume_yes]
```

-   `prefix <prefix>`: Clean up all the entities with the specified prefix (virtual machines, flash drives, networks).
-   `assume_yes`: Quietly agree to erase suggested virtual entities.

> Running `testo clean` with no argmuments will result in cleaning up all the entities without any prefix. Manually user-created entities won't be touched.

**Return value**: 0

## Tests queueing algorithm

By default, `testo` interpreter schedules to run all the tests in the `.testo` files passed with the `input` argument. However, you can establish filters to specify the tests you want (`test_spec`) or don't want (`exclude`) to run. These command-line arguments can be passed an arbitrary number of times and in any order you want. The tests queueing algorithm resembles a pipeline and looks like this:

1. First, all the tests in `.testo` files are found, thus forming a set of test names.
2. The set of names are then passed into the first filter. If the filter is `test_spec`, then only the tests with the names matching `wildcard_pattern` remain after applying the filter. If the filter is `exclude`, then only the tests with the names **not** matching `wildcard_pattern` remain after applying the filter.
3. The resulting set of tests is passed into the next filter. The filter processing algorithm remains the same.
4. When all the filters are processed, the remaining tests are queued to run.

### Wildcard pattern format

| Syntax | Meaning |
| --- | --- |
| `*` | Any number of any characters |
| `?` | Any single charachter  |
| `\` | Escape symbol |
| `[abc]` | Any of the characters specified in the brackets brackets |
| `[!abc]` | Any of the characters except for those specified in the brackets |
| <code>(abc&#124;c)</code> | Any of the character sequences listed in the parenthesis |
