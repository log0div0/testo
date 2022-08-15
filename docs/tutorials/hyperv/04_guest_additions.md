# Guide 4. Guest Additions

## What you're going to learn

In this guide you're going to learn how to install and exploit Testo guest additions.

## Preconditions

1. Testo Framework is installed.
2. Hyper-V is installed.
3. [Ubuntu server 16.04](https://releases.ubuntu.com/16.04/ubuntu-16.04.7-server-amd64.iso) image is downloaded and located here: `C:\iso\ubuntu_server.iso`. The location may be different, but in this case the test scripts have to be adjusted accordingly.
4. The Host has the Internet access.
5. Testo guest additions iso image is downloaded and located in the same folder as Ubuntu Server 16.04 iso-image.
6. (Recommended) Testo-lang [syntax highlight](/en/docs/getting_started/getting_started#setting-up-testo-lang-syntax-highlighting) for Sublime Text 3 is set up.
7. (Recommended) [Guide 3](03_nat) is complete.

## Introduction

In the last guide we've learned how to automate the Ubuntu Server 16.04 installation and got ourselved acquainted with the basic actions in Testo-lang mimicking the human behaviour. The approach we've used is good from two points of view:
1. Allows to test the SUT in exactly the same way as the end user (real human) would've tested it.
2. Doesn't require any additional agents running in virtual machines.

However, there is a downside to this approach as well: confusing and not intuitive console command execution. You must have noticed it by yourself: to run a bash-command `ping 8.8.8.8 -c5` like a human, we had to do this:

```testo
type "ping 8.8.8.8 -c5 && echo Result is &?"; press Enter
wait "Result is 0" timeout 10s
```

Of course, that's not what you want to see in your test scripts. Much simpler would be to use something like this:

```testo
exec bash "ping 8.8.8.8 -c5"
```

and just rely on the exit code of the bash script.

To overcome this problem, the Testo Framework comes with the Guest Additions iso-image. The Guest Additions support various guest operating systems, including, but not limited to: Linux-based Ubuntu and CentOS, as well as Windows 7 and 10. When the Guest Additions are installed in the virtual machine, you unlock new high-level actions in your test scripts: [`exec`](/en/docs/lang/actions_vm#exec) - execute a script (bash, python or cmd) on the virtual machine, [`copyto`](/en/docs/lang/actions_vm#copyto) - copy files from the Host to the guest and [`copyfrom`](/en/docs/lang/actions_vm#copyfrom) - copy files from the guest to the Host.

Guest Additions are recommended for installation in a virtual machine in two cases:
1. If the virtual machine is secondary and you don't care much about it.
2. If the guest additions wouldn't affect the software under test behaviour.

To sum everything up, you should install Guest Addiitons in the virtual machines in most cases and discard them only when the additions are not installable or just undesired.

## What to begin with?

The guest additions require special support from the guest virtual machine. A Hyper-V virtual machine must support Hyper-V sockets for the guest additions to work. Hyper-V sockets are the special hypervisor technology that creates an additional channel between the Host and the guest.

Windows guest systems has such support enabled by default. But the Linux-based operating systems situation is not that clear: some distributions have this feature by default as well (CentOS), other ditributions can have that support installed, while the rest need the coresponding kernel module compiled from sources.

In this guides we're using Ubuntu Server 16. Hyper-V sockets support is implemented in the `hv_sock` kernel module, but it is disabled by default. We can see for ourselves that this module is not loaded in the kernel and we can't insert it either:

![hv_sock_missing](/static/docs/tutorials/hyperv/04_guest_additions/hv_sock_missing.png)

Well, this module is not included in the default Ubuntu Server 16 kernel. But it is included on the `linux-azure` kernel, which can be installed with the `apt` manager.

After the module is inserted in the kernel, all is left to do is to mount the testo-guest-additions iso image and install the .deb package from that iso-image.

So, we have the following plan:

1. Install the `linux-azure` package.
2. Reboot the machine.
3. Insert the `hv_sock` module in the kernel.
4. Plug the guest additions iso image into the virtual DVD-drive.
5. Mount the plugged iso image into the guest file system (if not done automatically).
6. Run the guest additions installation (may differ depending on the guest OS).

> Steps 1-3 may differ depending on the operating system. As previously mentioned, have the neccessary support enabled by default. The other may have completely different ways to enable this support. If you have any questions regarding the Hyper-V sockets support in your operating system - please, contact us on support@testo-lang.ru

> Keep in mind, that the plan above requires the Host to have the Internet access.

## Preparing Ubuntu Server

For starters, let's take care of the `linux-azure` package installation. We'll divide this task into simple steps:

1. Enter sudo-mode (`sudo su`).
2. Reset the DHCP-settings of the `eth0` NIC (which provides the Internet access). This steps requires an exmplanation. As we learned from the previous guide, Testo runs only tests that took some changes in them. The `ubuntu_installation` test hasn't changed since the end of the previous test and, therefore, it won't run again. Instead, the `my_ubuntu` machine would be restored into the state it was at the end of the `ubuntu_installation` test, when the corresponding snapshot was created. But the problem is that sometimes the network settings "freeze up" after a snapshot restoreation and so those settings need to be reset. This issue is relevant to the Hyper-V hypervisor.
3. Run `apt update` and make sure that the manager is up and running.
4. Run `apt install -y linux-azure`
5. Reboot the machine.
6. Log in and enter the sudo-mode once again.
7. Insert the now-available `hv_sock` and make sure that the module has been actually inserted.

All these steps will take place in the `ubuntu_prepare` test, which depends on the `ubuntu_installation` test. The `check_internet` test has no use for us now, so it can be deleted.


```testo
test ubuntu_prepare: ubuntu_installation {
	my_ubuntu {
		type "sudo su"; press Enter
		wait "password for my-ubuntu-login"; type "1111"; press Enter
		wait "root@my-ubuntu"

		abort "stop here"

		type "dhclient -r eth0 && dhclient eth0 && echo Result is $?"; press Enter

		type "clear && apt update && echo Result is $?"; press Enter
		wait "Result is 0"

		type "clear && apt install -y linux-azure && echo Result is $?"; press Enter
		wait "Result is 0" timeout 15m		

		type "reboot"; press Enter

		wait "login:" timeout 2m; type "my-ubuntu-login"; press Enter
		wait "Password:"; type "1111"; press Enter
		wait "Welcome to Ubuntu"

		type "sudo su"; press Enter;
		wait "password for my-ubuntu-login"; type "1111"; press Enter
		wait "root@my-ubuntu"

		type "clear && modprobe hv_sock && echo Result is $?"; press Enter;
		wait "Result is 0"

		type "clear && lsmod | grep hv"; press Enter
		wait "hv_sock"
	}
}
```

There are a couple of interesting moments worth mentioning in the script above:
1. Bash-commands are run with the chain `clear, command, echo Result is $?, wait "Result is $?"`. This chain can be translated as the following: clear the screen, execute a command, print the return code on the screen and check that this code is actually zero (which means everything is OK). The screen clearing have to be done to purge the output from the previous commands, because that output may interfere with the current command's execution.
2. To make sure that the module is properly inserted, we print all the inserted modules which have the `hv` characters in the,. If the `hv_sock` module was among those modules, then it would be displayed on the screen, where it would be "catched" with the `wait "hv_sock"` action.

> Don't worry just yet about the somewhat messy and cumbersome way we have to handle bash commands. First, with the guest additions installed, this process would be much more streamlined and convenient. Second, even if you had to execute bash commands without guest additions, this still could be simplified greatly with macros. We'll consider this technology in one of the future guides.

Let's try to run this test:

<Asset id="terminal1"/>

As you can see, everything went according to the plan. The test was staged so it wouldn't be run again without a good reason.

## Installing the Guest Additions

Now it's time to automate the guest additions installation. We'll place this task into yet another test `ubuntu_install_guest_additions` which depends on on the `ubuntu_prepare` test which we've just completed.

In the last guide we've learned about the `unplug dvd` action, which "ejects" the mounted iso-image from the DVD-drive of the virtual machine. Naturally, there is a `plug dvd` action in Testo-lang, allowing you to "insert" an iso-image to the DVD-drive. This action, however, takes an argument - a path to the iso-image you want to "insert".

```testo
test ubuntu_install_guest_additions: ubuntu_prepare {
	my_ubuntu {
		plug dvd "C:\\iso\\testo-guest-additions-hyperv.iso"
		abort "stop here"
	}
}
```

> Make sure that you downloaded the Hyper-V testo guest additions. Don't mix it up with the QEMU testo guest additions!

Try to run this script (don't forget the `--sto_on_fail` command line argument), wait for the breakpoint to trigger and open the virtual machine properties with the virt-manager. In the CDROM section you'll find the information about the iso-image you've just plugged.

![CDROM plugged](/static/docs/tutorials/hyperv/04_guest_additions/plugged_cdrom.png)

Now we need to mount the DVD-drive into the Ubuntu filesystem and install the .deb package.

```testo
test ubuntu_install_guest_additions: ubuntu_prepare {
	my_ubuntu {
		plug dvd "C:\\iso\\testo-guest-additions-hyperv.iso"

		type "mount /dev/cdrom /media"; press Enter
		wait "mounting read-only"
		type "clear && dpkg -i /media/*.deb && echo Result is $?"; press Enter;
		wait "Result is 0"
		type "clear && umount /media && echo Result is $?"; press Enter;
		wait "Result is 0"
		sleep 2s
		unplug dvd
	}
}
```


Take a note, that we used a new action: [`sleep`](/en/docs/lang/actions_vm#sleep). This aciton works exactly how you'd exect it to work: just waits unconditionally for the specified amount of time.

<Asset id="terminal2"/>

And so the guest additions are installed. They are now up and running, so let's try them out.

## Trying out the guest additions

To try out the guest additions, let's create a new test which is going to be a child to the test `ubuntu_install_guest_additions`. With the guest additions installed, we are able to use some new high-level acitons. In this guide we're going to focus on the `exec` action. Now let's try to execute a bash script which prints "Hello world" to the stdout.

```testo
test ubuntu_guest_additions_demo: ubuntu_install_guest_additions {
	my_ubuntu {
		exec bash "echo Hello world"
	}
}
```

The result:

<Asset id="terminal3"/>

We can see that the bash script run successfully. As the matter of fact, the `exec` action is not limited to bash scripts execution. You could also run python srcipts (if Python interpreter is available in the guest system at all). Scripts could be multiline (just encase them in triple quotes).

```testo
test ubuntu_guest_additions_demo: ubuntu_install_guest_additions {
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

<Asset id="terminal4"/>

We will learn other actions that come with the guest additions installation in the future guides.

## Conclusion

Guest additions can significantly simplify the command execution inside virtual machines. They can be (and should be) installed every time it's possible for greater convinience.

At the end of this guide we've got the next tests tree.

<img src="/static/docs/tutorials/hyperv/04_guest_additions/tests_tree.svg"/>

You can find the complete test scripts for this guide [here](https://github.com/testo-lang/testo-tutorials/tree/master/hyperv/04%20-%20guest%20additioins).
