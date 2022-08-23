# Macros

Macros allow you to group up frequently used blocks of code into a named entity which you can call multiple times.

There are three types of macros in Testo-lang: macros with actions, macros with commands and macros with declarations. All macro types have the same header syntax and differ only with the body fulfillment.

A macro declaration has the following syntax:

```text
macro <name> ([arg1, arg2, ..., argn="default_value"]) {
	<macro_body>
}
```

Macros require a unique for all the macros `<name>`, which must be an identifier. Macros could take arguments, which could be referenced inside a macro's body. At the moment only string arguments are allowed. Arguments could also have default values. Inside default values [param referencing](Params.md#param-referencing) is available.

## Macros with actions

A macro with action is, quite simply, is a macro consisting solely of actions. The declaration for this type of macros looks like this:

```text
macro <name> ([arg1, arg2, ..., argn="default_value"]) {
	action1
	action2
	...
}
```

Inside macro's body all the actions (including macro calls), conditions and cycles are allowed.


## Example

```testo
# login macro waits for a prompt and performs a login attempt
# Has 2 arguments: a login and a password. Password argument has a default value
# which is calculated based on the "default_password" param.
# "default_password" must be defined for this to work
macro login(login, password="${default_password}") {
	wait "login:"; type "${login}"; press Enter
	wait "Password:"; type "${password}"; press Enter
	wait "Welcome to Ubuntu"
}

# unplug_nic macro shuts down the vm and unplugs the specified nic
# has one argmuent - nic_name
macro unplug_nic(nic_name) {
	shutdown
	unplug nic "${nic_name}"
}

test my_test {
	my_ubuntu {
		...
		unplug_nic("internet")
		start

		# we specify only one argument because
		# the root has the default_password password
		login("root")
		...
	}
}
```

## Macros with commands

A macro with commands is a macro with the body consisting of commands. You can check out the command syntax [here](Tests.md#commands-syntax). The declaration of such a macro looks like this:

```text
macro <name> ([arg1, arg2, ..., argn="default_value"]) {
	command1
	command2
	...
}
```

A command consists of two parts: a virtual entity (flash drive of virtual machine) name and an action (or a block of actions) which must be applied to this entity. The entity name can be represented two ways: as an indentifier (`some_id`) and as a string (`"some_id"`).

The string representation could be especially useful in some cases, because you can use reference macro arguments inside the strings. Using string entity name representation allows you to pass the virtual entities' names inside macros. Let's consider the following example:

```testo
macro copy_folder(first_machine, second_machine, flash_to_copy, src, dst) {
	"${first_machine}" {
		plug flash "${flash_to_copy}"
		exec bash """
			cp -r ${src} /media/flash/${src}
		"""
		unplug_flash "${flash_to_copy}"
	}

	"${second_machine}" {
		plug flash "${flash_to_copy}"
		exec bash """
			cp -r /media/flash/${src} "${dst}"
		"""
		unplug_flash "${flash_to_copy}"
	}
}
```

This macro's task is to copy a folder from one virtual machine to another with a flash drive. It is assumed that both virtual machines have the testo guest additions installed (they are required for the `exec bash` actions to work). The arguments for this macro are: virtual machines names, virtual flash drive name, source folder path and destination path.

Take notice, that the macro consists of commands, but the names of entities are parameterized, which means that this macro can be applied to any couple of virtual machines.

You can find the call example for this macro below.

> A macro must consist either solely of actions, or solely of commands. Macros with commands and actions mixed together are prohibited.

## Macros with declarations

A macro with declarations is a macro with the body consisting of declarations of tests, virtual machines, flash drives and networks. The declaration of such a macro looks like this:

```text
macro <name> ([arg1, arg2, ..., argn="default_value"]) {
	declaration_1
	declaration_2
	...
}
```

Where `declaration_i` can be either of the following:
- Test declaration;
- Virtual machine declaration;
- Virtual flash drive declaration;
- Virtual network declaration;
- A call of another macro with statements.

Declaring macros and params inside a macro is **prohibited**. It is also prohibited to use the `include` directive inside a macro.

Macros with declarations let you combine look-alike tests and even look-alike test benches inside a single piece of code. Such macros may come handy when you need to generate several equal (or almost equal) tests.

**Examples**

Let's consider a case when you need to create a couple of tests ("install OS" + "prepare OS") for the same OS, but with different architecture (32-bit and 64-bit). In this case you'd need two vritual machines (one for 32-bit OS and one for 64-bit OS) and four tests ("install OS 32-bit", "prepare OS 32-bit", "install OS 64-bit", "prepare OS 64-bit"). Let's also imagine that you need to give more RAM to the 64-bit OS. This task can easily be done with a macro with declarations:

```testo
macro generate_tests(bits, memory) {

	machine "vm_${bits}" {
		cpus: 2
		ram: "${memory}"
		iso: "${ISO_DIR}/os_${bits}.iso"
		disk main: {
			size: 5Gb
		}
	}

	test "vm_${bits}_install_os" {
		"vm_${bits}" {
			install_os("${bits}")
		}
	}

	test "vm_${bits}_prepare_os": "vm_${bits}_install_os" {
		"vm_${bits}" prepare()
	}

}
```

This macro allows you to parameterize not only the virtual bench (although it consists of only one virtual machine), but also the tests involving this bench. The virtual machine has its name, RAM amount and ISO-path parameterized (we assume that there are different ISO-images for different architectures). The tests have their names (and their parent's names) parameterized, as well as virtual machines' names inside the tests. The test `"vm_${bits}_install_os"` has the OS with corresponding architecture installation (we assume that the installation process is pretty similar and can be encapsulated inside a macro with actions `install_os()`). The `"vm_${bits}_prepare_os"` test just invokes the `prepare()` macro which performs the same actions to prepare the OS.

At the same time you can divide this macro into two: one for the virtual machines declaration and one for the tests declaration:

```testo
macro generate_vms(bits, memory) {

	machine "vm_${bits}" {
		cpus: 2
		ram: "${memory}"
		iso: "${ISO_DIR}/os_${bits}.iso"
		disk main: {
			size: 5Gb
		}
	}
}

macro generate_tests(bits) {

	test "vm_${bits}_install_os" {
		"vm_${bits}" {
			install_os("${bits}")
		}
	}

	test "vm_${bits}_prepare_os": "vm_${bits}_install_os" {
		"vm_${bits}_prepare_os" prepare()
	}

}
```

> Please keep in mind, that the macro declaration does not mean the actual tests and machines declaration. To do that you need to call the macro (see below).

## Macro call

A macro call can be either an action, command or a declaration - depending on the macro's body. If the macro contains commands, then it must be called as a command. If the macro contains actions, then it must be called as an action. If the macro contains declarations it must be called either at the global level (with all the other declarations), or inside another macro with declarations (see example 5 below).

> An attempt to call a macro in a wrong place will lead to an error. For example, an attempt to call a macro with declarations as an action is an error.

A macro call has the following syntax:

```text
<macro_name>([param1, param2, ...])
```

**Arguments**:

The number of passed arguments must not exceed the number of arguments in the corresponding macro declaration. If default arguments are used in macro declaration, you can omit one or more trailing arguments. Only string arguments are allowed.

**Example 1**

The example below demonstrates the concepts of the macro with actions call. Keep in mind, that a macro should only contain the actions that could be applied to the entity type for which this macro is called. This also is true for nested macro calls.

```testo
macro suits_vm() {
	press Right, Enter

	if (check "Hello") {
		type "World"
	}

	suits_fd_and_vm_with_ga()
}

# Suits a flash drive (in any case) and
# a virtal machine if it has the guest
# additions installed
macro suits_fd_and_vm_with_ga() {
	copyto "/some_file_on_host" "/some_file_on_guest"
}

macro suits_both() {
	print "Hello world"

	if (NOT DEFINED some_var) {
		abort "some var is not defined"
	}
}

machine some_vm {
	...
}

flash some_flash {
	...
}

test macro_call_usage_example {
	# Correct, if the some_vm virtual machine has the guest additions installed
	some_vm suits_vm()

	# Correct in any case. Copyto is always available to flash drives
	some_flash_drive suits_fd_and_vm_with_ga()

	# Correct
	some_vm suits_both()
	some_flash_drive suits_both()

	# Error: the "check" expression is not available for flash drives
	some_flash_drive suits_vm()
}
```

**Example 2**

In the example below you can see the macro with commands `copy_folder` usage demonstration. The macro itself is presented above.

```testo
machine client {
	...
}

machine server {
	...
}

machine gateway {
	...
}

machine firewall {
	...
}

flash copy_flash {
	...
}

test client_server_test {
	copy_folder("client", "server", "copy_flash", "/opt", "/opt")
}

test firewall_gateway_test {
	copy_folder("firewall", "gateway", "copy_flash", "/opt", "/opt")
}

test some_test {
	client {
		copy_folder() #Error: the macro contains commands, but we attempt to call it as an action
	}
}
```

**Example 3**

Let's check out the usage of a macro with declaration. As mentioned before, the macro declaration doesn't automatically mean the declaration of entities inside the macro. For the actual declarations you must call the macro:

```testo
macro generate_tests(bits, memory) {

	machine "vm_${bits}" {
		cpus: 2
		ram: "${memory}"
		iso: "${ISO_DIR}/os_${bits}.iso"
		disk main: {
			size: 5Gb
		}
	}

	test "vm_${bits}_install_os" {
		"vm_${bits}" {
			install_os("${bits}")
		}
	}

	test "vm_${bits}_prepare_os": "vm_${bits}_install_os" {
		"vm_${bits}" prepare()
	}

}

generate_tests("32", "4Gb")
generate_tests("64", "8Gb")

# Now you can reference the tests which have been generated by the macro
test some_test_for_32_bit_os: vm_32_prepare_os {
	vm_32 {
		print "Hello world"
	}
}

```

**Example 4**

You can also declare machines and tests separately:

```testo
macro generate_vms(bits, memory) {

	machine "vm_${bits}" {
		cpus: 2
		ram: "${memory}"
		iso: "${ISO_DIR}/os_${bits}.iso"
		disk main: {
			size: 5Gb
		}
	}
}

macro generate_tests(bits) {

	test "vm_${bits}_install_os" {
		"vm_${bits}" {
			install_os("${bits}")
		}
	}

	test "vm_${bits}_prepare_os": "vm_${bits}_install_os" {
		"vm_${bits}_prepare_os" prepare()
	}

}

# Error! Generating tests BEFORE the virtual machines generation
# means that the test will have an uknown virtual machine
generate_tests("32")


# This way is OK
generate_vms("32", "4Gb")
generate_tests("32")
```

**Example 5**

Inside a macro with declarations you can call another macros with declarations:

```testo
macro generate_vms(bits, memory) {

	machine "vm_${bits}" {
		cpus: 2
		ram: "${memory}"
		iso: "${ISO_DIR}/os_${bits}.iso"
		disk main: {
			size: 5Gb
		}
	}
}

macro generate_tests(bits, memory) {
	generate_vms("${bits}", "${memory}")

	test "vm_${bits}_install_os" {
		"vm_${bits}" {
			install_os("${bits}")
		}
	}

	test "vm_${bits}_prepare_os": "vm_${bits}_install_os" {
		"vm_${bits}_prepare_os" prepare()
	}
}

# Now this call is not a problem
# Because the tests generation now include
# the generation of the corresponding virtual machine
generate_tests("32", "4Gb")
```
