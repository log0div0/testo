# Guide 3. Guest Additions

## What you're going to learn

In this guide you're going to learn:
1. Test hierarchy basics.
2. Testo Guest Additions installation routine.

## Preconditions

1. Testo Framework is installed.
2. Virt manager is installed.
3. [Ubuntu server 16.04](https://releases.ubuntu.com/16.04/ubuntu-16.04.7-server-amd64.iso) image is downloaded and located here: `/opt/iso/ubuntu_server.iso`. The location may be different, but in this case the test scripts have to be adjusted accordingly.
4. Testo guest additions iso image is downloaded and located in the same folder as Ubuntu Server 16.04 iso-image.
5. (Recommended) Testo-lang [syntax highlight](/en/docs/getting_started/getting_started#setting-up-testo-lang-syntax-highlighting) for Sublime Text 3 is set up.
6. (Recommended) [Guide 2](02_ubuntu_installation) is complete.

## Introduction

In the last guide we've learned how to automate the Ubuntu Server 16.04 installation and got ourselved acquainted with the basic actions in Testo-lang mimicking the human behaviour. The approach we've used is good from two points of view:
1. Allows to test the SUT in exactly the same way as the end user (real human) would've tested it.
2. Doesn't require any additional agents running in virtual machines.

However, there is a downside to this approach as well: confusing and not intuitive console command execution. Indeed, to execute a bash command using a human-mimicking approach you'd need to do something like that:

```testo
type "command_to_execute";
press Enter
type "echo Result is $?"; press Enter
wait "Result is 0"
```

Of course, that's not what you want to see in your test scripts. Much simpler would be to use something like this:

```testo
exec bash "command_to_execute"
```

and just rely on the exit code of the bash script.

To overcome this problem, the Testo Framework comes with the Guest Additions iso-image. The Guest Additions support various guest operating systems, including, but not limited to: Linux-based Ubuntu and CentOS, as well as Windows 7 and 10. When the Guest Additions are installed in the virtual machine, you unlock new high-level actions in your test scripts: [`exec`](/en/docs/lang/actions_vm#exec) - execute a script (bash, python or cmd) on the virtual machine, [`copyto`](/en/docs/lang/actions_vm#copyto) - copy files from the Host to the guest and [`copyfrom`](/en/docs/lang/actions_vm#copyfrom) - copy files from the guest to the Host.

Guest Additions are recommended for installation in a virtual machine in two cases:
1. If the virtual machine is secondary and you don't care much about it.
2. If the guest additions wouldn't affect the software under test behaviour.

To sum everything up, you should install Guest Addiitons in the virtual machines in most cases and discard them only when the additions are not installable or just undesired.

## What to begin with?

To install the guest additions in an already installed guest OS, you need to perform several easy steps:
1. Plug the guest additions iso image into the virtual DVD-drive.
2. Mount the plugged iso image into the guest file system (if not done automatically).
3. Run the guest additions installation (may differ depending on the guest OS).

Let's go back to the test script we've developed in the Guide 2, in which we automated the Ubuntu Server 16.04 installation.

Let's begin with renaming the test `my_first_test` into something more appropriate. For example, `ubuntu_installation`.

```testo
test ubuntu_installation {
	my_ubuntu {
		start
		wait "English"
		...
		wait "login:" timeout 2m; type "my-ubuntu-login"; press Enter
		wait "Password:"; type "1111"; press Enter
		wait "Welcome to Ubuntu"
	}
}
```

Don't run the new script just yet, we'll do that in a jiffy. Right now let's proceed to the guest additions installation automation.

But first we need to learn what the tests [hiearchy](/en/docs/getting_started/test_policy) is. All the tests are organized in a tree based on the "testing from the simple to the complex" concept. Tests can have a "parent-children" connection, where the "simpler" test plays the role of the parent and the "more complex" test plays the role of the child. A test may have multiple parents and multiple children. A child is not going run until **all** his parents are completed successfully.  The most simple tests that don't depend on anything (and have no parents) are called **base** tests, otherwise a test is called **derived**. In our guides the `ubuntu_installation` test is an example of a base test.

The "child-parent" connection is created like this:

```testo
test guest_additions_installation: ubuntu_installation {
	my_ubuntu {
		abort "stop here"
	}
}
```

Let's make sure that a child test really depends on the parent test.

To do that, let's run our script file, but this time we're going to use a new command-line argument `--test_spec`. This argument specifies the tests we want to run (instead of running all of the tests). After the run we'll see something like this:

<Asset id="terminal1"/>

At the begin of the output we can see, that the Testo Framework is shceduled the `ubuntu_installation` test to run, despite the fact that we only asked to run the `guest_additions_installation` test. It happened because we want to run a child test, but the parent test hasn't run successfully just yet. And therefore, Testo Framework automatically queues the parent test first, and only after that - the child test.

But haven't we already installed Ubuntu successfully? We ended the previous guide at the moment when the Ubuntu Server was installed, the test ended, and the virtual machine state must had been staged.

In fact, in the previous guide our test had the name `my_first_test`, and now, with the new name, Testo Framework sees it as a brand new test, which has never run before.

At the end of the output we can see, that the parent test has been run successfully, and Testo proceeded to the child test run, but it failed (because of the `abort` action).

If we run Testo one more time with the same arguments, we will see a new picture:

<Asset id="terminal2"/>

This means that Testo Framework detected, that the `ubuntu_installation` test had already run successfully, and its virtual machine state had been staged. And so, instead of running the parent test again, Testo was able to just rollback the test bench into the state it was at the end of `ubuntu_installation` test.

## Installing the Guest Additions

Now it's time to automate the guest additions installation. In the last guide we've learned about the `unplug dvd` action, which "ejects" the mounted iso-image from the DVD-drive of the virtual machine. Naturally, there is a `plug dvd` action in Testo-lang, allowing you to "insert" an iso-image to the DVD-drive. This action, however, takes an argument - a path to the iso-image you want to "insert".

```testo
test guest_additions_installation: ubuntu_installation {
	my_ubuntu {
		plug dvd "/opt/iso/testo-guest-additions.iso"
		abort "stop here"
	}
}
```

Try to run this script (don't forget the `--sto_on_fail` command line argument), wait for the breakpoint to trigger and open the virtual machine properties with the virt-manager. In the CDROM section you'll find the information about the iso-image you've just plugged.

![CDROM plugged](/static/docs/tutorials/qemu/03_guest_additions/plugged_cdrom.png)

Now we need to mount the DVD-drive into the Ubuntu filesystem. Since we'd need the root privileges to do that, let's enter the sudo mode first.

```testo
test guest_additions_installation: ubuntu_installation {
	my_ubuntu {
		plug dvd "/opt/iso/testo-guest-additions.iso"

		type "sudo su"; press Enter;
		wait "password for my-ubuntu-login"; type "1111"; press Enter
		wait "root@my-ubuntu"

		abort "stop here"
	}
}
```

And now, finally, let's install the guest additions deb-package. Don't forget to umount the DVD-drive and "eject" the iso from the DVD-drive after that.

```testo
test guest_additions_installation: ubuntu_installation {
	my_ubuntu {
		plug dvd "/opt/iso/testo-guest-additions.iso"

		type "sudo su"; press Enter;
		wait "password for my-ubuntu-login"; type "1111"; press Enter
		wait "root@my-ubuntu"

		type "mount /dev/cdrom /media"; press Enter
		wait "mounting read-only"; type "dpkg -i /media/*.deb"; press Enter;
		wait "Setting up testo-guest-additions"
		type "umount /media"; press Enter;
		# Give a little time for the umount to do its job
		sleep 2s
		unplug dvd
	}
}
```

Take a note, that we used a new action: [`sleep`](/en/docs/lang/actions_vm#sleep). This aciton works exactly how you'd exect it to work: just waits unconditionally for the specified amount of time.

You may now remove the `abort` at the end of the test to complete it.

<Asset id="terminal3"/>

And so the guest additions are installed. They are now up and running, so let's try them out.

## Trying out the guest additions

To try out the guest additions, let's create a new test which is going to be a child to the test `guest_additions_installation`. With the guest additions installed, we are able to use some new high-level acitons. In this guide we're going to focus on the `exec` action. Now let's try to execute a bash script which prints "Hello world" to the stdout.

```testo
test guest_additions_demo: guest_additions_installation {
	my_ubuntu {
		exec bash "echo Hello world"
	}
}
```

The result:

<Asset id="terminal4"/>

We can see that the bash script run successfully. As the matter of fact, the `exec` action is not limited to bash scripts execution. You could also run python srcipts (if Python interpreter is available in the guest system at all). Scripts could be multiline (just encase them in triple quotes).

```testo
test guest_additions_demo: guest_additions_installation {
	my_ubuntu {
		exec bash """
			echo Hello world
			echo from bash
		"""
		# Double quotes require the escape symbol in one-line strings
		exec python2 "print('Hello from python2!')"
		exec python3 "print('Hello from python3!')"
	}
}
```

<Asset id="terminal5"/>

We will learn other actions that come with the guest additions installation in the future guides. For instance, `copyto` actions are considered in [guide 5](05_caching).

## Conclusion

At the end of this guide we've got the next tests tree.

<img src="/static/docs/tutorials/qemu/03_guest_additions/tests_tree.svg"/>

You can find the complete test scripts for this guide [here](https://github.com/testo-lang/testo-tutorials/tree/master/qemu/03%20-%20guest%20additions).
