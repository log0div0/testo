# Guide 8. Flash Drives

## What you're going to learn

In this guide you're going to learn about virtual flash drives in Testo Framework.

## Preconditions

1. Testo Framework is installed.
2. Virt manager is installed.
3. Host has the Internet access.
4. [Ubuntu server 16.04](https://releases.ubuntu.com/16.04/ubuntu-16.04.7-server-amd64.iso) image is downloaded and located here: `/opt/iso/ubuntu_server.iso`. The location may be different, but in this case the `ISO_DIR` command-line param has to be adjusted accordingly.
5. Testo guest additions iso image is downloaded and located in the same folder as Ubuntu Server 16.04 iso-image.
6. (Recommended) Testo-lang [syntax highlight](/en/docs/getting_started/getting_started#setting-up-testo-lang-syntax-highlighting) for Sublime Text 3 is set up.
7. (Recommended) [Guide 7](07_ping) is complete.

## Introduction

The last virtual infrastructure piece we haven't learn about yet is virtual flash drives. Virtual flash drives are suitable in two cases:

1. To transfer files between virtual machines without the network.
2. To copy files from the Host to a virtual machine (and vise versa) if you can't use the guest additions and the `copyto/copyfrom` actions.

There are two important virtual flash drives features:

1. At the end of each successful test, all the involved flash drives are staged (alongside with the virtual machines). The state then can be restored (just like virtual machines snapshots), so you can be sure that during the test run the flash drive is in the correct state.
2. Flash Drives attributes (and file checksums for the `folder` attribute) are a part of the test checksum. This means that Testo Framework will detect flash drives configuration modifications and re-run all the tests where the flash drives are referenced, if necessary.

There is an important **restriction** when using virtual flash drives: at the end of a test all the plugged flash drives **must be ungplugged**.

At the moment, there is also an another restriction: you can't plug two or more flash drives at the same virtual machine simultaneously.

In this guide we're going to learn all about the virtual flash drives and how to use them.

## What to begin with?

Let's assume that we need to transfer a file from the `client` machine to the `server`, and for some reason we don't want to use the network for that. In this case a virtual flash drive may come in handy. But first we need to declare it with the [`flash`](/en/docs/lang/flash) directive.

```testo
flash exchange_flash {
	fs: "ntfs"
	size: 16Mb
}
```

Flash drives declaration is similar to the virtual machines and networks declarations: the `flash` keyword, followed by a unique name and a set of attributes, some of which are mandatory. There are two mandatory attributes for flash drives:

1. `fs` - filesystem type. In our case it's the `ntfs`;
2. `size` - the flash drive size. 16 Megabytes are more than enough.

Just like the case with virtual machines, a virtual flash drive declaration doesn't mean its actual creation. The flash drive is created when the test with a reference to this flash drive runs for the first time.

Let's create a new test in which we're going to transfer a file from `client` to `server`.

```testo
# Take a note, that the test is inherited not from
# the test_ping, but from the client_prepare and
# server_prepare tests.
# Which means that test_ping and exchange_files_with_flash
# both lay on the samy tests tree level
test exchange_files_with_flash: client_prepare, server_prepare {
	client {
		# Create a file to be transferred to the server
		exec bash "echo \"Hello from client!\" > /tmp/copy_me_to_server.txt"

		plug flash exchange_flash
		sleep 5s
		exec bash """
			mount /dev/sdb1 /media
			cp /tmp/copy_me_to_server.txt /media
			umount /media
		"""

		unplug flash exchange_flash
	}

	server {
		plug flash exchange_flash
		sleep 5s
		exec bash """
			mount /dev/sdb1 /media
			cp /media/copy_me_to_server.txt /tmp
			umount /media
			cat /tmp/copy_me_to_server.txt
		"""

		unplug flash exchange_flash
	}
}
```

As you can see, this test is pretty much straightforward: we create a file on the `client` and then plug the `exchange` flash drive into the machine (mimicking a real human plugging a real flash drive into a real computer).

Since there is no auto-mounting for flash drives in Ubuntu Server, we need to mount the `exchange` flash drive manually. After the inserting, we need to wait for several seconds to give the OS the time to react to the new device. The inserted flash drive is visible inside the OS as the `/dev/sdb` device, from which we need the first partition `/dev/sdb1`. Mount this partition, copy the file to the flash drive, unmount the device and safely remove the flash drive.

On the `server` side, we perform the same plug/mount actions and then print the file to the stdout. Don't forget to unplug the flash drive at the end of the test.

Let's try to run this script (all tests must be cached before the run, otherwise the following stdout may differ):

<Asset id="terminal1"/>

As we can see, at the beginnig of the test the virtual flash drive is created. The file transferring works as expected, and in the end we can read a Hello from the client printed on the server.

As we already mentioned, at the end of the test the flash drive state is staged, so if some new files are written to the flash drvie in the children tests, these files won't show up in the parent tests.

It is really important to unplug all the plugged flash drives at the end of the test. If this is not done - an error will be generated when staging the test bench.

## Copying files from the Host using flash drives

There is another case for using virtual flash drives in Testo: transferring files from the Host to the virtual machines when the guest additions are not available or not preferable.

To do this we need to specify the `folder` attribute in the flash drive declaration:

```testo
flash exchange_flash {
	fs: "ntfs"
	size: 16Mb
	folder: "./folder_to_copy"
}
```

This attribute contains a path to the Host **folder** which you want to copy on the flash drive. You can specify a relative path, in which case the starting point for the path resolving is the directory where the test script file is located. You can't specify a single file, it must be a folder.

> The folder itself isn't copied at the root point in the flash drive filesystem. Rather all the folder's contents are copied at the `/` in the flash drive. You may consider the **folder** attribute as a path to the mount point from which all the contents are copied to the flash drive.

Let's assume, that for some reason we can't use a `copyto` action to copy the `rename_net.sh` script to the `client` machine. To solve this problem, we're going to use a flash drive to do the same thing.

Create the folder `folder_to_copy` in the same directory where the `hello_world.testo` is located and copy `rename_net.sh` to `folder_to_copy`. Ok, now all we need to do is to adjust the `client_prepare` test a little:

```testo
test client_prepare: client_unplug_nat {
	client {
		plug flash exchange_flash
		sleep 5s
		exec bash """
			mount /dev/sdb1 /media
			cp /media/rename_net.sh /opt/rename_net.sh
			umount /media
		"""
		unplug flash exchange_flash
		exec bash """
			chmod +x /opt/rename_net.sh
			/opt/rename_net.sh 52:54:00:00:00:aa server_side
			ip a a 192.168.1.2/24 dev server_side
			ip l s server_side up
			ip ad
		"""
	}
}
```

Run the script:

<Asset id="terminal2"/>

All the tests passed successfully, so we can be sure that the `rename_net.sh` was copied inside the virtual machine just as planned.

A couple of interesting notes:

1. The `exhange` flash drive was re-created. The reason is that its configuration has been changed (we added the `folder` attribute). If the configuration had stayed intact, the flash drive would have been restored to the snapshot `initial`.
2. The `test_ping` test was scheduled to run, even though we didn't touch it. It happened because we've changed the `client_prepare` test, which is a parent to the `test_ping` test.
3. Files in the directory specified in the `folder` attribute, are a part of the checksum for any test which has a `plug` action with the flash drive.  So if you try to change the `folder_to_copy` contents (just modify a little the `rename_net.sh` file), then the `client_prepare` test will be re-run. We suggest you take a look at that by yourself.

## Copying files from the virtual machine using a flash drive

So we've covered transferring files into a virtual machine from the Host, using both guest additions and flash drives. But what about transferring files from the virtual machine into the Host?

Of course, if the virtual machine has the guest additions installed, the transferring is easily made with the `copyfrom` action (we suggest you to learn about this action by yourself). But what if a virtual machine doesn't have the guest additions installed? In this case we can use virtual flash drives.

Starting from the version 2.1.0 of Testo-lang, you can address flash drives in commands, just like virtual machines.

There are several actions you can apply to virtual flash drives, that are also applicable to virtual machines. For instance, you can call the [`copyto`](/en/docs/actions_fd#copyto)(copy files from the Host to the flash drive) and [`copyfrom`](/en/docs/actions_fd#copyfrom)(copy files from the flash drive to the Host) actions with virtual flash drives as subjects. The `copyto` and `copyfrom` actions for virtual machines require the guest additions to be installed in them, but at the same time these actions work always when applyed to virtual flash drives. The only thing you need to keep in mind is that the flash drive must not be plugged in any virtual machine when calling `copyto`/`copyfrom` actions.

So let's imagine, that at the end of the `exchange_files_with_flash` test we want to copy the `copy_me_to_server.txt` file from the flash drive to the Host. This file is already placed in the `exchange_flash` flash drive since we've put it there during the test. So the only thing left to do is to call the `copyfrom` action for the `exchange_flash` flash drive.

```
test exchange_files_with_flash: client_prepare, server_prepare {
	client {
		# Create a file to be transferred to the server
		exec bash "echo \"Hello from client!\" > /tmp/copy_me_to_server.txt"

		plug flash exchange_flash
		sleep 5s
		exec bash """
			mount /dev/sdb1 /media
			cp /tmp/copy_me_to_server.txt /media
			umount /media
		"""

		unplug flash exchange_flash
	}

	server {
		plug flash exchange_flash
		sleep 5s
		exec bash """
			mount /dev/sdb1 /media
			cp /media/copy_me_to_server.txt /tmp
			umount /media
			cat /tmp/copy_me_to_server.txt
		"""

		unplug flash exchange_flash
	}

	exchange_flash {
		copyfrom "/copy_me_to_server.txt" "./copy_me_to_server.txt"
	}
}
```

We can see at the end of the test the new command that addresses not a virtual machine - but a flash drive. The `copyfrom` action extracts the `copy_me_to_server.txt` file from the flash drive to the Host. So after the test run we can check out the file's contents on the Host:

<Asset id="terminal3"/>

There are two things that you need to keep in mind when using commands with flash drives:
1. The flash drive must not be plugged in any virtual machine during  `copyfrom`/`copyto` actions.
2. You should specifiy the full destination path in `copyfrom`/`copyto` actions. The requirement is the same as for the `copyto`/`copyfrom` actions in virtual machines.

We suggest you to learn about the `copyto` action for flash drives by yourself. As an exersice you may modify the `client_prepare` test in such a manner, that the copying of the `./folder_to_copy` folder would take place inside the test itself, not at the flash drive creation moment.

> The last task (copying a file from the virtual machine to the Host) will stay in this guide only and won't go out to the future guides.

## Conclusions

Virtual flash drives, alongside with virtual machines and networks, are all the virtual infrastructure entities available in Testo Framework. You can use virtual flash drives to transfer files between virtual machines or between the Host to virtual machines, if for some reason the guest additions are not suitable for you.

The tests hierarchy looks like this at the moment:

<img src="/static/docs/tutorials/qemu/08_flash/test_hierarchy.svg"/>

You can find the complete test scripts [here](https://github.com/testo-lang/testo-tutorials/tree/master/qemu/08%20-%20flash).
