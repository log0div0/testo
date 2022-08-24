# Tutorial 5. Caching

## What you're going to learn

In this guide you're going to learn about tests caching in Testo-lang.

## Introduction

Testo Framework is based on the concept "if you want to do something with a virtual machine - do it inside a test". On the one hand, it gives you a huge benefit: you can always run the test scripts "from the scratch" (without deploying the test bench and with no need for setting anything up). All you need is the test script files and iso-images (and maybe hard disk images, which will learn about a bit later). Therefore you can easily store the test scripts in any version control system with all its benefits. On the other hand, there is a downside to this approach: you have to put **all** the actions in the test scripts: even preparatory actions, aimed just to set the system up (OS installation, for example). Applying all those actions can take quite some time.

Obviously, running all the tests from the very beginning all the time would be very time consuming - the test bench deploying and setup could take much more time, than the actual meaningful test runs.

And because of that, the Testo Framework puts a lot of effort into saving as much time as possible when running tests. The main concept of time saving is to avoid runs for those tests, that actually don't need these runs.

The general idea is this: when tests are run for the first time (when there is no virtual machines created) - they are run from the top to bottom, from the very beginning. Every successful test is cached. Caching means the following:

1. For all the virtual machines and flash drives involved in the test, snapshots are created, staging their states at the end of the test (only if there is no `no_snapshots` attribute is specified, which we will learn about a bit later);
2. A set of metadata is created for the test. The metadata is stored on the disk and helps tracking cache consistency for the test.

If a test fails for some reason, no cache is created for it (naturally) and all its children tests are marked as failed by default.

The first complete tests run could take a while. You can draw a parallel between the first run and compilation of a big project from the source. After a time-consuming first complete compilation, all the new builds usually take lesser time, thanks to the incremental compilation (when only the object files with modified sources are recompiled). Testo Framework adopts a similar approach.

With the second run, Testo first validates the cache for the already successfully-run tests. A lot of [factors](../../reference/Tests.md#validating-the-test-cache) are taken into consideration, the main of which are:

1. Have the test scripts been modified or not (not significant changes are not considered);
2. Have the configurations for the virtual machines or flash drives involved in the test been modified or not;
3. Have the files involved in `copyto` and `plug dvd` been modified or not.

If the cache is considered valid, the test is not going to be run again. If all the tests have the valid cache, no tests are run at all (just like with the incremental compilation, when no sources have been modified). However, if the test cache is invalidated, the test **and all its children** are scheduled to run.

A shceduled to run test depends on the run results of its parents. Since all the successfully-run tests have the snapshots for all the virtual machines and flash drives, Testo Framework can restore their states and continue from the place the virtual infrastructure was staged.

To sum it up, we get the following picture:

1. First tests run takes the maximum amount of time, since **all** tests are run, even the most basic ones: OS installations, setting up the network settings and so on;
2. At the other runs, only the modified tests (where something significant changed) are actually run. Since the "preparatory" tests are mostly stable, they're likely to remain cached and not going to be run again.

## What to begin with?

In the last guide we've encountered a situation when all the tests were cached and wouldn't run again even after we replaced some string constants with param references. To understand why it happened we need to consider an example.

Let's take a look at our test hierarchy which we've developed during the last guide.

![tests tree](imgs/tests_tree.svg)

First we need to run all the tests to make sure they are all cached. If any tests weren't cached, then let them complete, and run the Testo again with the same arguments.

In the end you'll see the next output:

<Asset id="terminal1"/>

And now let's experiment a little with out test scripts and see what's going to happen with the cache.

First let's try to modify the actions in the `guest_additions_demo` test:

```testo
test guest_additions_demo: guest_additions_installation {
	my_ubuntu {
		# Modified bash script
		exec bash """
			echo Modified Hello world
			echo from bash
		"""
		# Double quotes require the escape symbol in one-line strings
		exec python3 "print(\"Hello from python3!\")"
	}
}
```

Now run the script and check out the output:

<Asset id="terminal2"/>

Pretty much as expected, the test modification resulted in the cache loss for the test. However, you can see that only the child-test `guest_additions_demo` was run. That happened because the `guest_additions_installation` test still had the valid cache, so Testo was able to restore the `my_ubuntu` virtual machine state from the `guest_additions_installation` snapshot, which had been created earlier during the run of the corresponding test.

Adding empty lines, tabs and comments doesn't affect the cache validity. Try to add or remove several empty lines or comments and see for yourself.

Now let's move on to the `guest_additions_installation` test. In this test we're going to use a param reference in the string specifying the deb-package.

```testo
param guest_additions_pkg "*.deb"
test guest_additions_installation: ubuntu_installation {
	my_ubuntu {
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
}
```

After running the script one more time we'll see this:

<Asset id="terminal3"/>

Which means that all tests are cached, despite that we've clearly modified one of the tests! We encountered the same situation during the previous guide. So what's the deal?

The thing is, when calculating checksums for tests cache, Testo takes into consideration only the **final** string values, after all the param references are resolved. Before we changed the string (by inserting the param reference), the action'd looked like `type "dpkg -i /media/*.deb"`. After the modificaiton and the param reference resolving, the action still looks exactly the same. And that's the reason why the cache is considered valid, despite visible modificaiton. That's what we saw in the last guide.

But let's change the param value:

```testo
param guest_additions_pkg "testo-guest-additions.deb"
test guest_additions_installation: ubuntu_installation {
	my_ubuntu {
		...
	}
}
```

Run the script again. If you don't want to accept the cache loss manually every time, you may want to use the `--assume_yes` command line argument.

<Asset id="terminal4"/>

Naturally, the `guest_additions_installation` test lost its cache, since the `guest_additions_pkg` value had changed. It is also worth noticing that the child-test `guest_additions_demo` lost its cache as well, even though we hadn't touched it in any way.

## File checksums in `copyto` and `plug dvd` actions

In the guide 3 we mentioned, that the guest additions unlock a few new high-level actions, including the `copyto` action, which allow to copy files from the Host system to virtual machines in your test scripts. Let's take a look at this action.

Let's assume that we need to copy a small text file inside the virtual machine. Create a file in the same folder where the `hello_world.testo` script is located.

The test itself needs to be modified.

```testo
test guest_additions_demo: guest_additions_installation {
	my_ubuntu {
		# Modified bash script
		exec bash """
			echo Modified Hello world
			echo from bash
		"""
		# Double quotes require the escape symbol in one-line strings
		exec python3 "print(\"Hello from python3!\")"

		copyto "./testing_copyto.txt" "/tmp/testing_copyto.txt"
		exec bash "cat /tmp/testing_copyto.txt"
	}
}
```

<Asset id="terminal5"/>

The `copyto` action takes two arguments: the file we want to copy (since it is located near the test script file itself, we can specify the relative path `./`) and the **full** abosulute path **including the destination file name**, specifying where we want to put our file on the virtual machine filesystem.

Run this script and make sure it is cached.

Now comes an important moment. As previously mentioned, when calculating the test checksums, Testo takes into account files involved in the test. The checksums are calculated following the next algorithm:

1. If the file size is less than 1 megabyte, then the checksum is calculated based on the **contents** of the file;
2. If the file size is equal or larger than 1 megabyte, then the checksum is calculated based on the **last modified** timestamp of the file;
3. If a folder is being copied, then the first two rules are applied to each file in the folder individually, the sizes of the files inside the folder are not summed up.

Let's make sure that the algorithm works as expected. Since our file `testing_copyto.txt` is less than 1 megabyte, its checksum is calculated based on its contents. You may change the last modified timestamp and see for yourself that the test remained cached. Changing the contents will immediatly result in cache loss.

You also may create a big file (greater than 1 megabyte) and make sure that its checksum is calculated based on its last modified timestamp.

> The same checksum calculation alrorithm is applied to the iso-images mentioned in `plug dvd` actions, and to the virtual flash drives `folder` attribute (if there is any). Virtual flash drives are explained in the future guides.

> There are other factors involved in test checksum calculations. For example, virtual machines and flash drives configurations. You can find all these factors in the [documentation](../../reference/Tests.md#validating-the-test-cache). In particular, iso-images in the `iso` attribute of virtual machines also affect the cache validation.

> There is a possibility to adjust the threshold of the file size that changes the checksum alrogithm. It is done with the `--content_cksum_maxsize` command line argument.

## Manual cache reset

Of course, there is a way to force reset the test cache. You can do that with the `--invalidate` command line argument, which has the same format as `--test_spec` and lets you specify a test name matching pattern.

For example, if you want to reset the cache for all the tests related to guest additions, just run the command below:

<Asset id="terminal6"/>

## Conclusions

Caching is an important part of the Testo Framework and is aimed to save as much time as possible for the second and following test runs. Cache lets Testo run only the tests that need to be run (has been modified in some way).