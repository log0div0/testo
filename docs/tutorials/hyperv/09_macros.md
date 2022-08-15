# Guide 9. Macros

## What you're going to learn

In this guide you're going to learn about:
1. Macros in Testo-lang.
2. How to distribute your scripts among several .testo files.

## Preconditions

1. Testo Framework is installed.
2. Hyper-V is installed.
3. [Ubuntu server 16.04](https://releases.ubuntu.com/16.04/ubuntu-16.04.7-server-amd64.iso) image is downloaded and located here: `C:\iso\ubuntu_server.iso`. The location may be different, but in this case the `ISO_DIR` command-line param has to be adjusted accordingly.
4. Testo guest additions iso image is downloaded and located in the same folder as Ubuntu Server 16.04 iso-image.
5. The Host has the Internet access.
6. (Recommended) Testo-lang [syntax highlight](/en/docs/getting_started/getting_started#setting-up-testo-lang-syntax-highlighting) for Sublime Text 3 is set up.
7. (Recommended) [Guide 7](07_ping) is complete.

## Introduction

During the previous guides we've written quite a lot of code, some of which is almost exact copy-paste. For example, you could've noticed, that the `server_install_ubuntu` and `client_install_ubuntu` tests look almost exactly the same, and the only difference is that they have different virtual machines and params. All the actions are basically the same.

It is a natural desire to clean the mess up a little bit and "hide" similar lines of code in some sort of encapsulating language constructs. In regular programming languages it may be done with funcitons, procedures and so on, but in Testo-lang it is done with macros.

In Testo-lang a macro basically is a named **action**, **command** or **declarations** block. A macro call is also an **action**, a **command** or a **declaration** (depending on the macro type). With macros you can group up similar pieces of code into named blocks, so that your scripts are more streamlined and easier to maintain. Macros can take arguments (and default-valued arguments as well) which can be referenced inside the macro body as usual params.

And of course you can distribute your scripts among different files. Script files are then linked with each other with `include` directives, which we're going to see in action in this guide.

## What to begin with?

It easy to notice, that we have a lot of similar preparatory actions for the `client` and `server` machines: Ubuntu Server installation and preparations, guest additions installation. To be honest, it is just a lot of copy-paste code, which looks ugly. But with a little effort we can clean this up. Using macros, of course.

Let's consider the OS installation. Right now the server Ubuntu Server installation test looks like this:

```testo
test server_install_ubuntu {
	server {
		start
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

		wait "Is this time zone correct?" timeout 2m; press Enter
		wait "Partitioning method"; press Enter
		wait "Select disk to partition"; press Enter
		wait "Write the changes to disks and configure LVM?"; press Left, Enter
		wait "Amount of volume group to use for guided partitioning"; press Enter
		wait "Force UEFI installation?"; press Left, Enter
		wait "Write the changes to disks?"; press Left, Enter
		wait "HTTP proxy information" timeout 3m; press Enter
		wait "How do you want to manage upgrades" timeout 6m; press Enter
		wait "Choose software to install"; press Enter
		wait "Installation complete" timeout 30m;

		unplug dvd; press Enter
		wait "login:" timeout 2m; type "${server_login}"; press Enter
		wait "Password:"; type "${default_password}"; press Enter
		wait "Welcome to Ubuntu"
	}
}
```

For the `client`, the picture is not much different: `client` instead of `server`, different `hostname` and `login` values. Aside from that, everything looks exactly the same. This is the perfect candidate for our first macro.

Let's declare our first [macro](/en/docs/lang/macro) and name it `install_ubuntu`. The declaration must be placed at the global level, where all the other declaraions go.

```testo
macro install_ubuntu(hostname, login, password) {
	start
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

	wait "Is this time zone correct?" timeout 2m; press Enter
	wait "Partitioning method"; press Enter
	wait "Select disk to partition"; press Enter
	wait "Write the changes to disks and configure LVM?"; press Left, Enter
	wait "Amount of volume group to use for guided partitioning"; press Enter
	wait "Force UEFI installation?"; press Left, Enter
	wait "Write the changes to disks?"; press Left, Enter
	wait "HTTP proxy information" timeout 3m; press Enter
	wait "How do you want to manage upgrades" timeout 6m; press Enter
	wait "Choose software to install"; press Enter
	wait "Installation complete" timeout 30m;

	unplug dvd; press Enter
	wait "login:" timeout 2m; type "${login}"; press Enter
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

<Asset id="terminal1"/>

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

<Asset id="terminal2"/>

## Preparing the OS

Now let's handle the `server_prepare` and `client_prepare` tests. These two tests look very similar, so, therefore, we can apply a macro too:


```testo
macro prepare_ubuntu(hostname, login, password = "${default_password}") {
	# Enter sudo mode
	type "sudo su"; press Enter
	wait "password for ${login}"; type "${password}"; press Enter
	wait "root@${hostname}"

	# Reset the eth0 NIC to prevent any issues with it after the rollback
	type "dhclient -r eth0 && dhclient eth0 && echo Result is $?"; press Enter

	# Check that apt is OK
	type "clear && apt update && echo Result is $?"; press Enter
	wait "Result is 0"

	# Install linux-azure package
	type "clear && apt install -y linux-azure && echo Result is $?"; press Enter
	wait "Result is 0" timeout 15m		

	# Reboot and login
	type "reboot"; press Enter

	wait "login:" timeout 2m; type "${login}"; press Enter
	wait "Password:"; type "${password}"; press Enter
	wait "Welcome to Ubuntu"

	# Enter sudo once more
	type "sudo su"; press Enter;
	wait "password for ${login}"; type "${password}"; press Enter
	wait "root@${hostname}"

	# Load the hv_sock module
	type "clear && modprobe hv_sock && echo Result is $?"; press Enter;
	wait "Result is 0"

	type "clear && lsmod | grep hv"; press Enter
	wait "hv_sock"
}
```

Now the `client_prepare` and `server_prepare` tests look streamlined and compact as well:

```testo
test client_prepare: client_install_ubuntu {
	client prepare_ubuntu("${client_hostname}", "${client_login}")
}

test server_prepare: server_install_ubuntu {
	server prepare_ubuntu("${server_hostname}", "${server_login}")
}
```

But we can do even more than that: we can make the `prepare_ubuntu` macro more elegant. We can find out that this macro has two identical set of actions just to enter the `sudo` mode:

```testo
type "sudo su"; press Enter;
wait "password for ${login}"; type "${password}"; press Enter
wait "root@${hostname}"
```

This set of actions can also be placed in a macro of its own:

```testo
macro enter_sudo(hostname, login, password) {
	type "sudo su"; press Enter;
	wait "password for ${login}"; type "${password}"; press Enter
	wait "root@${hostname}"
}
```

The `prepare_ubuntu` macro will look like this:

```testo
macro prepare_ubuntu(hostname, login, password = "${default_password}") {
	# Enter sudo mode
	enter_sudo("${hostname}", "${login}", "${password}")

	# Reset the eth0 NIC to prevent any issues with it after the rollback
	type "dhclient -r eth0 && dhclient eth0 && echo Result is $?"; press Enter

	# Check that apt is OK
	type "clear && apt update && echo Result is $?"; press Enter
	wait "Result is 0"

	# Install linux-azure package
	type "clear && apt install -y linux-azure && echo Result is $?"; press Enter
	wait "Result is 0" timeout 15m		

	# Reboot and login
	type "reboot"; press Enter

	wait "login:" timeout 2m; type "${login}"; press Enter
	wait "Password:"; type "${password}"; press Enter
	wait "Welcome to Ubuntu"

	# Enter sudo once more
	enter_sudo("${hostname}", "${login}", "${password}")

	# Load the hv_sock module
	type "clear && modprobe hv_sock && echo Result is $?"; press Enter;
	wait "Result is 0"

	type "clear && lsmod | grep hv"; press Enter
	wait "hv_sock"
}
```

As you can see, the arguments for the `enter_sudo` macro call are calculated based on the `prepare_ubuntu` argument values. You should also keen in mind, that you're only allowed to call macros with actions inside the `prepare_ubuntu` (not with commands and not with declarations). The `enter_sudo` macro contains actions, so everything is OK.

The `enter_sudo` macro is a very basic and short macro. It could come hanfy in future: inside tests and other macros.

Let's make sure that the tests are still cached:

<Asset id="terminal3"/>

## Guest additions installation


Let's move on and turn our focus to the guest additions installation. Clearly this is also a perfect candidate to implement a macro: the guest additions installation looks exactly the same for both virtual machines.

```testo
param guest_additions_pkg "testo-guest-additions*"

macro install_guest_additions() {
	plug dvd "${ISO_DIR}\\testo-guest-additions-hyperv.iso"

	type "mount /dev/cdrom /media"; press Enter
	wait "mounting read-only"
	type "clear && dpkg -i /media/${guest_additions_pkg} && echo Result is $?"; press Enter;
	wait "Result is 0"
	type "clear && umount /media && echo Result is $?"; press Enter;
	wait "Result is 0"
	sleep 2s
	unplug dvd
}
```

In this macro we can see a situation worth noticing and explaining. Inside the macro body we reference the `ISO_DIR` and `guest_aditions_pkg`, even though they are not present in the argument list. But we still can reference them, because of the param resolving [algorithm](/en/docs/lang/param#resolving-algorithm):

1. When the reference is encountered inside a macro, Testo checks whether a macro argument or a global param is being referenced. If a macro argument is referenced, the algorithm returns its value and the resolving stops.
2. If a global param (including params specified with the `--param` command line arguments) is referenced, its value is returned and the resolving stops. In our case the algorithm stops at this step when referencing `${ISO_DIR}` and `${guest_additions_pkg}`.
3. If nothing was found, an error is generated.

Guest additions installation tests are now also much smaller and easier to read:

```testo
test server_install_guest_additions: server_install_ubuntu {
	server install_guest_additions()
}

test client_install_guest_additions: client_install_ubuntu {
	client install_guest_additions()
}
```

And still everything is cached:

<Asset id="terminal4"/>

## A macro for executing bash commands

That last thing in this guide we'll improve with macros is bash command executions. As you could've noticed, until the guest additions was installed we'd been forced to execute bash commands with the following chain: `clean; bash_command && echo "Result is $?"; wait "Result is 0"`.

This chain does its work, but it's tedious to write this every time. But we can create a macro encapsulating all this logic:

```testo
macro exec_bash_command(command) {
	type "clear && ${command} && echo Result is $?"; press Enter
	wait "Result is 0"
}
```

With this macro the way to execute a bash command without the guest additions becomes every simple: just call the `exec_bash_command` macro and pass the bash command as the argument. Let's update the `prepare_ubuntu` macro:

```testo
macro prepare_ubuntu(hostname, login, password = "${default_password}") {
	# Enter sudo mode
	enter_sudo("${hostname}", "${login}", "${password}")

	# Reset the eth0 NIC to prevent any issues with it after the rollback
	exec_bash_command("dhclient -r eth0 && dhclient eth0")
	# Check that apt is OK
	exec_bash_command("apt update")
	# Install linux-azure package
	exec_bash_command("apt install -y linux-azure")

	# Reboot and login
	type "reboot"; press Enter

	wait "login:" timeout 2m; type "${login}"; press Enter
	wait "Password:"; type "${password}"; press Enter
	wait "Welcome to Ubuntu"

	# Enter sudo once more
	enter_sudo("${hostname}", "${login}", "${password}")

	# Load the hv_sock module
	exec_bash_command("modprobe hv_sock")

	type "clear && lsmod | grep hv"; press Enter
	wait "hv_sock"
}
```

Looks better, doesn't it? Although right now it's not that perfect: the `exec_bash_command("apt install -y linux-azure")` macro call wouldn't work. The reason is that the `wait "Result is 0"` action inside the macro would wait the output only for 1 minute (default timeout). Which is clearly not enough considering that `apt install -y linux-azure` takes much more time than that. So what can we do?

We can exploit another great Testo-lang feature, allowing us to *use strings instead of several language tokens*. In particular, with Testo-lang you can specify the `timeout` for `wait` actions with strings, not the special time-specifier:

```testo
wait "Hello world" timeout 5m
wait "Hello world" timeout "5m" #The same thing
```

You may be wondering: "How exactly would this help me?". Well, the help comes with the fact that you can reference params inside strings!

```testo
wait "Hello world" timeout "${time_to_wait}"
```

Now to updating the `exec_bash_command`:

```testo
macro exec_bash_command(command, time_to_wait = "1m") {
	type "clear && ${command} && echo Result is $?"; press Enter
	wait "Result is 0" timeout "${time_to_wait}"
}
```

We just added the second argument specifying the time to wait. We also gave it the default value so we wouldn't have to pass it with every call of the macro.

The final version of the `prepare_ubuntu` macro is this:

```testo
macro prepare_ubuntu(hostname, login, password = "${default_password}") {
	# Enter sudo mode
	enter_sudo("${hostname}", "${login}", "${password}")

	# Reset the eth0 NIC to prevent any issues with it after the rollback
	exec_bash_command("dhclient -r eth0 && dhclient eth0")
	# Check that apt is OK
	exec_bash_command("apt update")
	# Install linux-azure package
	exec_bash_command("apt install -y linux-azure", "15m")

	# Reboot and login
	type "reboot"; press Enter

	wait "login:" timeout 2m; type "${login}"; press Enter
	wait "Password:"; type "${password}"; press Enter
	wait "Welcome to Ubuntu"

	# Enter sudo once more
	enter_sudo("${hostname}", "${login}", "${password}")

	# Load the hv_sock module
	exec_bash_command("modprobe hv_sock")

	type "clear && lsmod | grep hv"; press Enter
	wait "hv_sock"
}
```

Let's also apply the `exec_bash_command` macro to the `install_guest_additions`:

```testo
macro install_guest_additions() {
	plug dvd "${ISO_DIR}\\testo-guest-additions-hyperv.iso"

	type "mount /dev/cdrom /media"; press Enter
	wait "mounting read-only"

	exec_bash_command("dpkg -i /media/${guest_additions_pkg}")
	exec_bash_command("umount /media")
	
	sleep 2s
	unplug dvd
}
```

Run the tests again and this time you'll see that tha cache has actually been invalidated:

<Asset id="terminal5"/>

The reason is that the `prepare_ubuntu` macro had the action at its beginning:

```testo
type "dhclient -r eth0 && dhclient eth0 && echo Result is $?";
```

which transformed into the next action after we'd applied the `exec_bash_command` macro:

```testo
type "clear && dhclient -r eth0 && dhclient eth0 && echo Result is $?";
```

Since the action had been modified, Testo invalidated the cache for all tests calling the `ubuntu_prepare` macro. This includes `client_prepare` and `server_prepare` tests. And all their children as well.

## Macros with declarations

In Testo-lang it is also possible to use macros with declarations. This topic is covered [here](16_macro_with_declarations).

## `include` directive

We've done a good job sorting out the scripts and grouping up actions and commands into macros. The `macro.testo` file now looks much neater and cleaner. But there is still a nagging feeling of a mess: in the same file we have the entities declarations, preparatory tests and "actual" tests. For now this may not give us a lot of headache, but the more the script size grows, the harder it would be to navigate through it. So let't try to split our code into several linked files.

Instead of one big file `macro.testo` we're going to have several designated files: `declarations.testo` (put all the declarations there, including params), `macros.testo` (macros) and `tests.testo` (you've guessed it). Of course, this is only one of the possible ways to distribute the code, you may group up the code to your liking.

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

1. Run the "terminal" script file: `testo run tests.testo --stop_on_fail --param ISO_DIR C:\iso`.
2. Run the whole folder with tests: `testo run .\ --stop_on_fail --param ISO_DIR C:\iso`.

## Conclusions

Macros and the `include` directive are a great way to simplify and streamline your test scripts. The more code you develop, the more important it is to distribute your code among different files, othwerwise you risk to turn your tests into a one big mess.

And if you do everything carefully, you may even avoid cache losses, since the caching in Testo doesn't care much for macros, but rather for the actions in them.

You can find the complete test scripts [here](https://github.com/testo-lang/testo-tutorials/tree/master/hyperv/09%20-%20macros).
