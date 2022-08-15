# Tests declaration

Tests are declared starting with the `test` keyword. The declaration has the following syntax:

```text
[[
	<attr1>: <value1>
	<attr2>: <value1>
	<attr3>: <value1>
]]
test <test_name>[: test_parent1, test_parent2, ...]
{
	command1
	command2
	...
}
```

> Tests can also be defined inside macros. See [here](macros#macros-with-declarations) for more information.

A test declaration has an optional header with a set of attributes (enclosed in brackets `[]`) followed by a newline, a keyword `test`, a tests-unique name `test_name`, an optional parental tests list and a test body. The test body is a list of commands.

Test name can be either an identifier or a string. If a string is used, the value inside the string must be convertible to a identifier. Inside the string [param referencing](param#param-referencing) is available.

Test attributes has the same syntax as virtual machines', flash drives' and networks' attributes. The syntax is described [here](machine).

All the attributes are **optional**. There are only two possible test attributes (all optional) at the time:
- `no_snapshots` - Type: boolean. If `true`, no hypervisor snapshots will be created for the virtual resources (virtual machines and flash drives) participating in the test. For more information see [here](/en/docs/getting_started/test_policy#tests-without-hypervisor-snapshots). Default value is `false`.
- `description` - Type: string. Test description. Will be placed in a json-file, which is generated if the `json_report` command line argument is specified.

If the test depends on the successful results of some other tests, you should specify those tests in parental tests list. For example, a test with network configuration probably depends on a test with an operating system installation.

## Commands syntax

A test basically is a list of commands. A command has the following syntax:

```text
<vm_name | flash_drive_name> <action>
```

A command consists of two parts: a virtual entity's (machine or flash drive) name and a body (which could be an action, an [if-statement](if) or a [for-statement](for)) to be applied to the entity. A block of actions is also considered an action, in which case the command looks like this:

```text
<vm_name | flash_drive_name> {
	action1
	action2
	action3; action4; action5
	action6;
	{
		action7; action8
	}
}
```

> For Hyper-V you can use *only* commands with virtual machines. Commands with virtual flash drives are not available.

Actions must be separated with a newline or a semicolon (`;`).

The list of [actions](actions_vm) available for virtual machines differs from the list of [actions](actions_fd) available for flash drives.

A [macro](macro) with commands call is also considered a command.

## Virtual entity's name parameterization

In commands, the entity name can be represented two ways: as an indentifier (`client`, `my_flash`) and as a string (`"client"`, `"my_flash"`). Both ways are equal.

But at the same time the string representation could be useful in some cases, because you can use the [param](#param-referencing) inside the strings. This is especially useful if you want to pass the virtual entities' names inside macros.

> You should keep in mind, that some actions that are available for virtual machine, aren't available for flash drives. And therefore, there is a possibility that after the param resolving the command can turn up syntactically invalid (see example below).

**Examples**

```testo
# OK, if "client" is a virtual machine
client type "hello world"

# The same as above
"client" type "Hello world"

# Correct in any case
# since the "print" action is applicable both to VMs and flash drives
my_flash_drive print "Hello world"

# Same as above
"my_flash_drive" print "Hello world"

# Correct only if the "entity_name" value
# corresponds to one of the virtual machines' names
"${entity_name}" type "Hello world"
```

## Test running concepts

A test run is a four-step process:

- Validating the test cache.
- Preparing the running environment.
- Applying commands.
- Staging the running environment.

### Validating the test cache

Caching is an important technology in the Testo Framework. The main goal of this mechanism is to save your time by not running the tests that aren't needed to be run.

The first time a test is run successfully, a cache is created for this test.

When trying to run a cached test, its cache's consistency is evaluated. If nothing sufficient has changed in the test's script, the cache is considered valid and the test is not actually run. Othwervise the cache is lost, the test and all its children (recursively) are queued to run. Below you can see a complete list of all checks to be performed when evaluating the cache's validity.

- Do any of the test's parents have an invalid cache?
- Have the commands changed?
- Has the test's header changed?
- Have the values for the params referenced in the tests changed?
- Do all the referenced virtual machines have the [valid](machine#virtual-machines-caching) cache of theirs configurations?
- Do all the referenced virtual flash drives have the [valid](flash#virtual-flash-drives-caching) cache of theirs configurations?
- Have the checksums for the files and folders being copied to with the `copyto` actions changed?
- Have the checksums for the iso-files being plugged with the `plug dvd` actions changed?

If there is at least one "yes" answer to these questions, the chache is considered lost and the test is queued for the run. All the children are queued as well.

> You could always invalidate the cache manually with the `invalidate <wildcard match>` command line argument. All the children will lose the cache as well.

> There are two ways to check the files' cache consistency for the `copyto` and the `plug dvd` actions. If the file's size is less then 1 MB, then the file's integrity is evaluated based on its contents. Othwervise the evaluation is based on the Last modified timestamp of the file. You can adust the threshold of the evaluating mode changes with the `content_cksum_maxsize` command line argument.

If the cache is valid, then the test is not actually run and the Testo interpreter goes to the next test.

### Preparing the running environment

If the test is queued for a running, then all the virtual machines and virtual flash drives have to be restored into the states they need to be for the test to run:

- All the virtual machines, that were not previously mentioned in any parental test (if there is any) are created. After creation they will stay powered off, so you need to call the `start` action to activate them.
- If the parental tests don't have a `no_snapshots: true` attribute, then Testo  restores all the required snapshots for the virtual machines and the flash drives and reverts them to the states they were at the end of the parental tests.
- If the parental tests have a `no_snapshots: true` attribute, then the virtual resources (VMs and flash drives) from them don't have the hypervisor snapshots, and their state can't be restored. In this case Testo searches the tests hierarchy up for a "anchor" test (a test with hypervisor snapshots) and restores the states for the virtual machines and the virtual flash drives they were at the end of the "anchor" test. After that, all the intermediate parental tests are run, so the virtual machines and flash drives are reverted to the apropriate state.

### Applying commands

Applying commands is consequitive interpreting the actions to the virtual machines and flash drives mentioned at the beginning of a command.

If any action fails, the whole test is considered failed, and Testo moves on to the next test (only when no `--stop_on_fail` command line attribute is specified). If the failed test has any children, they are also considered failed by default.

### Staging the running environment

After a successful test run, Testo stages the bound virtual machines and flash rives in the state they are in at the end of the test. If the test doesn't have a `no_snapshots: true` attribute, then hypervisor snapshots for all the virtual machines and flash drives are created. The test's cache is updated.
