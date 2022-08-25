# Tutorial 9. Macros

## What you're going to learn

In this tutorial you're going to learn about:
1. Macros in Testo-lang.
2. How to distribute your scripts among several .testo files.

## Introduction

During the previous guides we've written quite a lot of code, some of which is almost exact copy-paste. For example, you could've noticed, that the `server_install_ubuntu` and `client_install_ubuntu` tests look almost exactly the same, and the only difference is that they have different virtual machines and params. All the actions are basically the same.

It is a natural desire to clean the mess up a little bit and "hide" similar lines of code in some sort of encapsulating language constructs. In regular programming languages it may be done with funcitons, procedures and so on, but in Testo-lang it is done with macros.

In Testo-lang a macro basically is a named **action**, **command** or **declaration** block. A macro call is also an **action**, **command** or **declaration** (depending on the macro type). With macros you can group up similar pieces of code into named blocks, so that your scripts are more streamlined and easier to maintain. Macros can take arguments (and default-valued arguments as well) which can be referenced inside the macro body as usual params.

And of course you can distribute your scripts among different files. Script files are then linked with each other with the `include` directive, which we're going to see in action in this tutorial.

## What to begin with?

It easy to notice, that we have a lot of similar preparatory actions for the `client` and `server` machines: Ubuntu Server installation, guest additions installation, unplugging NIC with the Internet access. To be honest, it is just a lot of copy-paste code, which looks ugly. But with a little effort we can clean this up. Using macros, of course.

Let's consider the OS installation. Right now the server Ubuntu Server installation test looks like this:

```testo
test server_install_ubuntu {
	server {
		start
		wait "English"
		press Enter
		# The actions can be separated with a newline
		# or a semicolon
		wait "Install Ubuntu Server"; press Enter;
		wait "Choose the language";	press Enter
		wait "Select your location"; press Enter
		wait "Detect keyboard layout?";	press Enter
		wait "Country of origin for the keyboard"; press Enter
		wait "Keyboard layout"; press Enter
		#wait "No network interfaces detected" timeout 5m; press Enter
		wait "Primary network interface"; press Enter
		wait "Hostname:" timeout 5m; press Backspace*36; type "${server_hostname}"; press Enter
		wait "Full name for the new user"; type "${server_login}"; press Enter
		wait "Username for your account"; press Enter
		wait "Choose a password for the new user"; type "${default_password}"; press Enter
		wait "Re-enter password to verify"; type "${default_password}"; press Enter
		wait "Use weak password?"; press Left, Enter
		wait "Encrypt your home directory?"; press Enter

		#wait "Select your timezone" timeout 2m; press Enter
		wait "Is this time zone correct?" timeout 2m; press Enter
		wait "Partitioning method"; press Enter
		wait "Select disk to partition"; press Enter
		wait "Write the changes to disks and configure LVM?"; press Left, Enter
		wait "Amount of volume group to use for guided partitioning"; press Enter
		wait "Write the changes to disks?"; press Left, Enter
		wait "HTTP proxy information" timeout 3m; press Enter
		wait "How do you want to manage upgrades" timeout 6m; press Enter
		wait "Choose software to install"; press Enter
		wait "Install the GRUB boot loader to the master boot record?" timeout 10m; press Enter
		wait "Installation complete" timeout 1m;

		unplug dvd; press Enter
		wait "${server_hostname} login:" timeout 2m; type "${server_login}"; press Enter
		wait "Password:"; type "${default_password}"; press Enter
		wait "Welcome to Ubuntu"
	}
}
```

For the `client`, the picture is not much different: `client` instead of `server`, different `hostname` and `login` values. Aside from that, everything looks exactly the same. This is the perfect candidate for our first macro.

Let's declare our first [macro](../../reference/Macros.md) and name it `install_ubuntu`. The declaration must be placed at the global level, where all the other declaraions go.

```testo
macro install_ubuntu(hostname, login, password) {
	start
	wait "English"
	press Enter
	# The actions can be separated with a newline
	# or a semicolon
	wait "Install Ubuntu Server"; press Enter;
	wait "Choose the language";	press Enter
	wait "Select your location"; press Enter
	wait "Detect keyboard layout?";	press Enter
	wait "Country of origin for the keyboard"; press Enter
	wait "Keyboard layout"; press Enter
	#wait "No network interfaces detected" timeout 5m; press Enter
	wait "Primary network interface"; press Enter
	wait "Hostname:" timeout 5m; press Backspace*36; type "${hostname}"; press Enter
	wait "Full name for the new user"; type "${login}"; press Enter
	wait "Username for your account"; press Enter
	wait "Choose a password for the new user"; type "${password}"; press Enter
	wait "Re-enter password to verify"; type "${password}"; press Enter
	wait "Use weak password?"; press Left, Enter
	wait "Encrypt your home directory?"; press Enter

	#wait "Select your timezone" timeout 2m; press Enter
	wait "Is this time zone correct?" timeout 2m; press Enter
	wait "Partitioning method"; press Enter
	wait "Select disk to partition"; press Enter
	wait "Write the changes to disks and configure LVM?"; press Left, Enter
	wait "Amount of volume group to use for guided partitioning"; press Enter
	wait "Write the changes to disks?"; press Left, Enter
	wait "HTTP proxy information" timeout 3m; press Enter
	wait "How do you want to manage upgrades" timeout 6m; press Enter
	wait "Choose software to install"; press Enter
	wait "Install the GRUB boot loader to the master boot record?" timeout 10m; press Enter
	wait "Installation complete" timeout 1m;

	unplug dvd; press Enter
	wait "${hostname} login:" timeout 2m; type "${login}"; press Enter
	wait "Password:"; type "${password}"; press Enter
	wait "Welcome to Ubuntu"
}
```

We're going to put all the actions, necessary to install an Ubuntu Server, in the macro body. Take a note that the macro has no virtual machine references: it's just a bunch of actions without any specific application.

`install_ubuntu` have three arguments: `hostname`, `login` and `password`, which are referenced inside the macro body in `type` actions: `type "${hostname}"`, `type "${login}"` and `type "${password}"`. Of course, the resulting argument values depend on the values passed with the macro call.

Now our tests `server_install_ubuntu` and `client_install_ubuntu` look like this:

```testo
test server_install_ubuntu {
	server install_ubuntu("${server_hostname}", "${server_login}", "${default_password}")
}

test client_install_ubuntu {
	client install_ubuntu("${client_hostname}", "${client_login}", "${default_password}")
}
```

Neat and clean, isn't it? Keep in mind, that the declared macro is a macro with **actions**, not a macro with **commands**. Since the macro contains actions, a call of this macro is also considered an action.

Let's try to run it!

![](imgs/terminal1.svg)

And so what are we seeing? The tests remained cached, even though we'd changed the base tests quite a lot (seemingly). However, the thing is, when tests checksums are being calculated, Testo Framework doesn't care much for macros: it just "unfolds" the macro body and places the actions instead of the macro call. Since the actions in the tests hadn't **actually** changed (we just moved them into the macro, which is not a significant change), the test checksum remained intact, and therefore the tests are still cached.

Well, the installation tests are now very compact and neat, but, as a matter of fact, we can make them even smaller! We can see, that the password arguments in both tests are the same: `"${default_password}"` (we doesn't need diffetent passwords in the machines, we're OK with the same default one). As you've probably guessed, we can give the `password` macro argument a default value. It is done like this:

```testo
param default_password "1111"
macro install_ubuntu(hostname, login, password = "${default_password}") {
	start
	wait "English"
	press Enter
	...
```

Take a note that the default value for `password` is resolved based on the param `default_password` value. The `default_password` param must be declared beforehand.

We don't need to pass the `password` argument with the macro calls anymore. The tests now look super-neat:

```testo
test server_install_ubuntu {
	server install_ubuntu("${server_hostname}", "${server_login}")
}

test client_install_ubuntu {
	client install_ubuntu("${client_hostname}", "${client_login}")
}
```

Try to run the script now, and you'll find out that the tests are still cached. The reason is the same: only the "unfolded" macros are taken into consideration when test checksums are calculated.

![](imgs/terminal2.svg)

## Guest additions installation

Let's move on and turn our focus to the guest additions installation. Clearly this is also a perfect candidate to implement a macro: the guest additions installation looks exactly the same for both virtual machines.

```testo
param guest_additions_pkg "testo-guest-additions*"

macro install_guest_additions(hostname, login, password="${default_password}") {
	plug dvd "${ISO_DIR}/testo-guest-additions.iso"

	type "sudo su"; press Enter;
	# Take a note that you may reference params in any part of the string
	wait "password for ${login}"; type "${password}"; press Enter
	wait "root@${hostname}"

	type "mount /dev/cdrom /media"; press Enter
	wait "mounting read-only"; type "dpkg -i /media/${guest_additions_pkg}"; press Enter;
	wait "Setting up testo-guest-additions"
	type "umount /media"; press Enter;
	# Give a little time for the umount to do its job
	sleep 2s
	unplug dvd
}
```

In this macro we can see a situation worth noticing and explaining. Inside the macro body we reference the `ISO_DIR` and `guest_aditions_pkg`, even though they are not present in the argument list. But we still can reference them, because of the param resolving [algorithm](../../reference/Params.md#resolve-order):

1. When the reference is encountered inside a macro, Testo checks whether a macro argument or a global param is being referenced. If a macro argument is referenced, the algorithm returns its value and the resolving stops. In our case the algorithms stops at this step when referencing `${hostname}`, `${login}` and `${password}`.
2. If a global param (including params specified with the `--param` command line arguments) is referenced, its value is returned and the resolving stops. In our case the algorithm stops at this step when referencing `${ISO_DIR}` and `${guest_additions_pkg}`.
3. If nothing was found, an error is generated.

Guest additions installation tests are now also much smaller and easier to read:

```testo
test server_install_guest_additions: server_install_ubuntu {
	server install_guest_additions("${server_hostname}", "${server_login}")
}

test client_install_guest_additions: client_install_ubuntu {
	client install_guest_additions("${client_hostname}", "${client_login}")
}
```

And still everything is cached:

![](imgs/terminal3.svg)

## unplug_nat macro

Next we turn our attention to the set of actions of the NIC `nat` unplugging. They too look almost identical in the `server` and `client` tests. Which calls for one more macro! At first look we could've implemented the macro somewhat like this:

```testo
macro unplug_nat(hostname, login, password="${default_password}") {
	shutdown
	unplug nic nat
	start

	wait "${hostname} login:" timeout 2m; type "${login}"; press Enter
	wait "Password:"; type "${password}"; press Enter
	wait "Welcome to Ubuntu"
}

...

test server_unplug_nat: server_install_guest_additions {
	server unplug_nat("${server_hostname}", "${server_login}")
}

test client_unplug_nat: client_install_guest_additions {
	client unplug_nat("${client_hostname}", "${client_login}")
}
```

And we could've been content with that, our goal is reached. But let's assume that we want to develop a macro for unplugging any NIC, not just the ones named `nat`.

To do that we need to parameterize the `unplug nic` action, and Testo-lang gives you such an opportunity. In Testo-lang some actions **optionally** can take strings as certain arguments, instead of the usual tokens. For example, the following actions are equal: `unplug nic nat` and `unplug nic "nat"`. The benefit of using strings instead of usual tokens is that you can use param referencing in such strings. So we may legitimately use the action `unplug nic "${nic_name}"`. The macro `unplug_nat` therefore can be generalized into `unplug_nic`, and the resulting script would be the following:

```testo
macro unplug_nic(hostname, login, nic_name, password="${default_password}") {
	shutdown
	unplug nic "${nic_name}"
	start

	wait "${hostname} login:" timeout 2m; type "${login}"; press Enter
	wait "Password:"; type "${password}"; press Enter
	wait "Welcome to Ubuntu"
}

...

test server_unplug_nat: server_install_guest_additions {
	server unplug_nic("${server_hostname}", "${server_login}", "nat")
}

test client_unplug_nat: client_install_guest_additions {
	client unplug_nic("${client_hostname}", "${client_login}", "nat")
}
```

This macro may be used to unplug any NIC, not just the `nat`.

In the [documentation](../../reference/Actions.md) you will find which actions allow strings arguments instead of regular tokens.

## process_flash macro

It's time now to deal with the flash drives handling.

If we take a closer look at the flash drives handling routine, we can recognize the following pattern:
1. Plug the flash into a virtual machine (`plug flash exchange_flash`).
2. Mount the flash into the guest OS filesystem (`exec bash "mount /dev/sdb1 /media"`).
3. Execute some bash-script (copy files).
4. Unmount the flash drive from the filesystem (`exec bash "umount /media"`).
5. Unplug the flash drive from the machine (`unplug flash exchange_flash`).

All these actions can be encapsulated in a macro. The flash drive name in the `(un)plug flash` action should be parameterized (like the NIC name in the previous example):

```testo
macro process_flash(flash_name, command) {
	plug flash "${flash_name}"
	sleep 5s
	exec bash "mount /dev/sdb1 /media"
	exec bash "${command}"
	exec bash "umount /media"
	unplug flash "${flash_name}"
}
```

The macro is named `process_flash` and has two arguments: a name for the flash drive to be plugged and a bash-script to be executed after the flash drive is mounted. The flash drive name is referenced in the actions `plug flash "${flash_name}"` and `unplug flash "${flash_name}"`.

Now we can conveniently use this macro when working with flash dirves and not worry about plugging, mounting and so on:

```testo
test client_prepare: client_unplug_nat {
	client {
		process_flash("exchange_flash", "cp /media/rename_net.sh /opt/rename_net.sh")

		exec bash """
			chmod +x /opt/rename_net.sh
			/opt/rename_net.sh 52:54:00:00:00:aa server_side
			ip a a 192.168.1.2/24 dev server_side
			ip l s server_side up
			ip ad
		"""
	}
}

...

test exchange_files_with_flash: client_prepare, server_prepare {
	client {
		# Create a file to be transferred to the server
		exec bash "echo \"Hello from client!\" > /tmp/copy_me_to_server.txt"
		process_flash("exchange_flash", "cp /tmp/copy_me_to_server.txt /media")
	}

	server {
		process_flash("exchange_flash", "cp /media/copy_me_to_server.txt /tmp")
		exec bash "cat /tmp/copy_me_to_server.txt"
	}
}
```

## A macro with commands

Up until this moment, we've been creating only macros with actions. The calls for these macros were done appropriately - as actions. However, in Testo-lang there is another type of macros as well: macros with commands. This type of macros can be extremely useful in some cases, which we shall now demonstrate.

Let's take a closer look a the `exchange_files_with_flash` test. In this test we copy a file between virtual machines with a flash drive. But let's consider this: basically, the routine to copy a file in such a manner woldn't really change if we put another flash drive and another pair of virtual machines (at least if the virtual machines had a Linux-based OS and testo guest additions installed). So, naturally, we get the temptation to incapsulate the file copy actions in some kind of macro. But the problem is, these actions (to copy a file) affect two virtual machines, not one. So a simple macro with actions (like what we've been doing in this tutorial) wouldn't do. We need a macro with commands:

```testo
macro copy_file_with_flash(vm_from, vm_to, copy_flash, file_from, file_to) {
	"${vm_from}" process_flash("${copy_flash}", "cp ${file_from} /media/$(basename ${file_from})")
	"${vm_to}" process_flash("${copy_flash}", "cp /media/$(basename ${file_from}) ${file_to}")
}
```

The new macro handles a file copying between any pair of virtual machines. It takes the following agruments:
- `vm_from` - source virtual machine name.
- `vm_to` - destination virtual machine name.
- `copy_flash` - name of the flash drive to be used.
- `file_from` - the source file path on the `vm_from`.
- `file_to` - the destination file path on the `vm_to`.

To make this macro possible, we used one more Testo-lang feature: you can use strings (instead of identifiers) to specify the virtual machines' and flash drives' names in commands. This means, that instead of `client` and `server` you can put `"client"` and `"server"`. We need this feature because of just one simple reason: inside strings we can reference params and macro arguments.

Inside the macro we write commands just like we're used to do in tests, except that the virtual entities are not some fixed VMs (`client` or `server`), but strings. And the strings' actual values are calculated based on the current macro agruments' `vm_from` and `vm_to` values. Therefore, this macro should work fine with any pair of virtual machines we pass with the arguments.

You can also see, that we used the `process_flash` macro (which we've developed earlier) inside the new macro body. The `process_flash` macro contains actions, and, hence, it must be called as an action (i.e. inside a command's body). The flash drive name argument for this macro is taken directly from the `copy_flash` agrument of the `copy_file_with_flash` macro.

To copy the file inside the flash drive's root directory, we extract the basename of the `file_from` file using the bash expression `$(basename ${file_from})`.

Now let's see how a calling for this new macro looks like:

```testo
test exchange_files_with_flash: client_prepare, server_prepare {
	client exec bash "echo \"Hello from client!\" > /tmp/copy_me_to_server.txt"
	copy_file_with_flash("client", "server", "exchange_flash", "/tmp/copy_me_to_server.txt", "/tmp/copy_me_to_server.txt")
	server exec bash "cat /tmp/copy_me_to_server.txt"
}
```

The test now consists of three commands:

1. The first command creates the file we want to copy.
2. The second command is the `copy_file_with_flash` macro call. The macro contains commands, so it must be called as a command.
3. The third command prints out the copyied file contents to stdout.

Of course, you can (and should) use this macro in the future to copy any file between two Linux-based virtual machines with any flash drive. All you need to do is to specify the virtual entities' names when calling the new macro.

## Macros with declarations

In Testo-lang it is also possible to use macros with declarations. This topic is covered the [tutorial 16](../16%20-%20macro%20with%20declarations).

## `include` directive

We've done a good job sorting out the scripts and grouping up actions and commands into macros. The `hello_world.testo` file now looks much neater and cleaner. But there is still a nagging feeling of a mess: in the same file we have the entities declarations, preparatory tests and "actual" tests. For now this may not give us a lot of headache, but the more the script size grows, the harder it would be to navigate through it. So let't try to split our code into several linked files.

Instead of one big file `hello_world.testo` we're going to have several designated files: `declarations.testo` (put all the declarations there, including params), `macros.testo` (macros) and `tests.testo` (you've guessed it). Of course, this is only one of the possible ways to distribute the code, you may group up the code to your liking.

Now we need to link the files up. In Testo-lang it is done with the `include` directive, which must be used at the same global level as declarations. You can't include a file inside a test, a vm configuration and so on. Only in between.

Let's get back to our files. `declarations.testo` doesn't depend on anyting - so no `include` there. `macros.testo` depends on `declarations.testo` because the `install_guest_addition` macro references the `default_password` and `guest_additions_pkg` params, which are declared there. So we include `declarations.testo` into `macros.testo`.

```testo
include "declarations.testo"

macro install_ubuntu(hostname, login, password = "${default_password}") {
	...
```

`tests.testo` depends on everything else. But since we've already included `declarations.testo` into `macros.testo`, we don't need to include it twice. So let's just include `macros.testo` into `tests.testo`:

```testo
include "macros.testo"

test server_install_ubuntu {
	server install_ubuntu("${server_hostname}", "${server_login}")
}
...
```

Now everything looks perfect: there is no duplicated code and everything is in its place. The only question remaining is how to run our tests now? There're two ways to do that:

1. Run the "terminal" script file: `sudo testo run tests.testo --stop_on_fail --param ISO_DIR /opt/iso`.
2. Run the whole folder with tests: `sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso`.

## Conclusions

Macros and the `include` directive are a great way to simplify and streamline your test scripts. The more code you develop, the more important it is to distribute your code among different files, othwerwise you risk to turn your tests into a one big mess.

And if you do everything carefully, you may even avoid cache losses, since the caching in Testo doesn't care much for macros, but rather for the actions in them.
