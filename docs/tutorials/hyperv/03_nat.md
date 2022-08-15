# Guide 3. Internet access in virtual machines

## What you're going to learn

In this guide you're going to learn:
1. Test hierarchy basics.
2. Learn about virtual networks and Network Interface Cards (NICs).
3. Learn how to provide the Internet access to virtual machines inside your test scripts.

## Preconditions

1. Testo Framework is installed.
2. Hyper-V is installed.
3. [Ubuntu server 16.04](https://releases.ubuntu.com/16.04/ubuntu-16.04.7-server-amd64.iso) image is downloaded and located here: `C:\iso\ubuntu_server.iso`. The location may be different, but in this case the test scripts have to be adjusted accordingly.
4. The Host has the Internet access.
5. (Recommended) Testo-lang [syntax highlight](/en/docs/getting_started/getting_started#setting-up-testo-lang-syntax-highlighting) for Sublime Text 3 is set up.
6. (Recommended) [Guide 2](02_ubuntu_installation) is complete.

## Introduction

Aside from virtual machines, there are two other virtual entities available in Testo-lang: virtual flash drives and virtual networks. Virtual flash drives are currently unavailable for Hyper-V, but virtual networks are OK to use, which we'll see in a moment.

Generally, virtual networks in Testo-lang could be used for 2 purposes: to link up virtual machines with each other and to provide the Internet access to a virtual machine. Linking up machines is explained in of of the next guides, so in this guide we're going to learn about the Internet access.

## What to begin with?

Let's begin with renaming the test `my_first_test` into something more appropriate. For example, `ubuntu_installation`.

```testo
test ubuntu_installation {
	my_ubuntu {
		start
		wait "Install Ubuntu Server"
		...
		wait "login:" timeout 2m; type "my-ubuntu-login"; press Enter
		wait "Password:"; type "1111"; press Enter
		wait "Welcome to Ubuntu"
	}
}
```

Don't run the new script just yet, we'll do that in a jiffy.

Right now the `my_ubuntu` machine doesn't have any NICs, so, obviously, it can't access the Internet. Let's create a test to check that it's true.

To do that first we need to learn what the tests [hiearchy](/en/docs/getting_started/test_policy) is. All the tests are organized in a tree based on the "testing from the simple to the complex" concept. Tests can have a "parent-children" connection, where the "simpler" test plays the role of the parent and the "more complex" test plays the role of the child. A test may have multiple parents and multiple children. A child is not going run until **all** his parents are completed successfully.  The most simple tests that don't depend on anything (and have no parents) are called **base** tests, otherwise a test is called **derived**. In our guides the `ubuntu_installation` test is an example of a base test.

The "child-parent" connection is created like this:

```testo
test check_internet: ubuntu_installation {
	my_ubuntu {
		abort "stop here"
	}
}
```

Let's make sure that a child test really depends on the parent test.

To do that, let's run our script file, but this time we're going to use a new command-line argument `--test_spec`. This argument specifies the tests we want to run (instead of running all of the tests). After the run we'll see something like this:

<Asset id="terminal1"/>

At the begin of the output we can see, that the Testo Framework is shceduled the `ubuntu_installation` test to run, despite the fact that we only asked to run the `check_internet` test. It happened because we want to run a child test, but the parent test hasn't run successfully just yet. And therefore, Testo Framework automatically queues the parent test first, and only after that - the child test.

But haven't we already installed Ubuntu successfully? We ended the previous guide at the moment when the Ubuntu Server was installed, the test ended, and the virtual machine state must had been staged.

In fact, in the previous guide our test had the name `my_first_test`, and now, with the new name, Testo Framework sees it as a brand new test, which has never run before.

At the end of the output we can see, that the parent test has been run successfully, and Testo proceeded to the child test run, but it failed (because of the `abort` action).

If we run Testo one more time with the same arguments, we will see a new picture:

<Asset id="terminal2"/>

This means that Testo Framework detected, that the `ubuntu_installation` test had already run successfully, and its virtual machine state had been staged. And so, instead of running the parent test again, Testo was able to just rollback the test bench into the state it was at the end of `ubuntu_installation` test.

## Makeing sure that there is no Internet

As I've mentioned earlier, the `my_ubuntu` machine doens't have the Internet acces just yet. AHow how can we make sure of that? We could try the `ping 8.8.8.8 -c5` bash command. But how can it be done? Well, we can do that just like any person would: type it on the keyboard:

```testo
test check_internet: ubuntu_installation {
	my_ubuntu {
		type "ping 8.8.8.8 -c5"; press Enter
		abort "stop here"
	}
}
```

Opening the machine in Hyper-V, we'll see the next result:

![Network_unreachable](/static/docs/tutorials/hyperv/03_nat/network_unreachable.png)

We, as humans, can clearly see there is no Internet. But what if we hadn't known that and we would've wanted to check that the **was** Internet access.

Of cource, the ping utility prints the information about the ICMP response packets (if there are any). We could've used the output itself, somewhat like this:

```testo
test check_internet: ubuntu_installation {
	my_ubuntu {
		type "ping 8.8.8.8 -c5"; press Enter
		wait "64 Bytes from 8.8.8.8"
		abort "stop here"
	}
}
```

But instead, I suggest more universal way to assert almost any command in Linux. It looks like this:

```testo
test check_internet: ubuntu_installation {
	my_ubuntu {
		type "ping 8.8.8.8 -c5 && echo Result is &?"; press Enter
		wait "Result is 0" timeout 10s
	}
}
```

What is going on here? We've exploited the Bash possibility to link up commands into a chain. In this particular case we're doing this "With the ping command returned successfully, pleasem execute also the `echo Result is &?`" command. Such an `echo` prints out the return code of the previously executed command (`ping`). If the ping was OK, then the `Result if 0` text would appear on the screen. Otherwise nothing would show up, or a return code would be non-zero.

Let's thy this script out:

<Asset id="terminal3"/>

And just to make sure that that the check would actually pass when the ping passed, let's change 8.8.8.8 for 127.0.0.1 (this ping should most definetely work fine):

```testo
test check_internet: ubuntu_installation {
	my_ubuntu {
		type "ping 127.0.0.1 -c5 && echo Result is &?"; press Enter
		wait "Result is 0" timeout 10s
	}
}
```

<Asset id="terminal4"/>

So the check does work. And with this type of check you can assert almost any bash command there is. In one of the future guides we'll move this code into a macro to make it more user-friendly.

> You can also execute commands inside an OS with the testo guest additions. We'll see how it's done in the next guide.

## Gaining the Internet access.

To connect your virtual machine to the Internet, first you need to declare a virtual network which will serve for this purpose. To declare a virtual network, the [`network`](/en/docs/lang/network) keyword is used:

```testo
network internet {
	mode: "nat"
}
```

Virtual network declaration is very similar to the virtual machine declarations: after the keyword `network`, a name for the network is followed (the name must be unique between all the virtual resources), followed by a set of attributes, from which only one is mandatory: `mode`. There are only two possible netowrk modes in Testo-lang at the moment: `nat` (the network is NATed to the default route of the Host, which usually means to the Internet) and `internal` (the network is isolated, this mode is used to link up virtual machines with each other).

After we've declared a network, it is time to link up our machine to this network. To do this, we need to add the `nic` attribute to `my_ubuntu` and attach this NIC to the `internet` network with the `attache_to` subattribute:

```testo
machine my_ubuntu {
	cpus: 1
	ram: 512Mb
	disk main: {
		size: 5Gb
	}
	iso: "C:\\iso\\ubuntu_server.iso"

	nic nat: {
		attached_to: "internet"
	}
}
```

The `nic` attribute must have a name (just like the `disk` attribute). The reason for that is that you can add several NICs to a virtual machine and unique names is the way to distinguish them (which will see in action in the next guide).

Aside from the name, you may to specify a bunch of subattributes, and only one of them is mandatory: `attached_to`, which specifies the virtual network name you want to attach the NIC to.

> Keep in mind that the `internet` network must be declared before referencing it in the `attached_to` subattribute.

## Adjusting the `ubuntu_installation` test

Let's run our test script:

<Asset id="terminal5"/>

If your tests are cached at the moment, you'll see that the `ubuntu_installation` test has lost its cache and needs to be re-run. It could seem strange, since we haven't modified the test itself, and its cache should've stayed intact. Hovewer, the cache validity depends on the involved virtual machines configurations integrity. Since we've changed the `my_ubuntu` configuration, all the tests referencing this virtual machine must be run again.

> Caching technology is considered thoroughly in one of the future guides

In any case, after a few minutes you're going to find out that the `ubuntu_installation` test now fails. The output gives us a hint, that Testo couldn't detect the "No network interfaces detected" text in 5 minutes.

Indeed, if you open the Hyper-V manager and check out the virtual machine screen, you'll see this:

![Hostname](/static/docs/tutorials/hyperv/03_nat/hostname.png)

The reason for this screen to appear is that we added the NIC to `my_ubuntu`, so now, naturally, the warning about not detecting any interfaces doesn't show up.

Let's comment up the line with waiting for this warning. But now 30 seconds may be not enough for the next text "Hostname" to appear, so let's enlarge this timeout to 5 minutes:

```testo
...
wait "Country of origin for the keyboard"; press Enter
wait "Keyboard layout"; press Enter
#wait "No network interfaces detected" timeout 5m; press Enter
wait "Hostname:" timeout 5m; press Backspace*36; type "my-ubuntu"; press Enter
wait "Full name for the new user"; type "my-ubuntu-login"; press Enter
wait "Username for your account"; press Enter
...
```

And run the script again.

If you don't have any issues with the proxy server (or if you don't use it at all), then the test script might fail one more time.

<Asset id="terminal6"/>

Instead of the screen with the time zone selection you'll see another screen:

![Timezone](/static/docs/tutorials/hyperv/03_nat/timezone.png)

The reason is that, thanks to the Internet access, Ubunstu Server installator now can detect your current timezone automatically. So let's change our script a little bit more:

```testo
...
wait "Re-enter password to verify"; type "1111"; press Enter
wait "Use weak password?"; press Left, Enter
wait "Encrypt your home directory?"; press Enter

#wait "Select your timezone" timeout 2m; press Enter
wait "Is this time zone correct?" timeout 2m; press Enter
wait "Partitioning method"; press Enter
...
```

Now all the tests should pass successfully. And the `check_internet` test along with it:

<Asset id="terminal7"/>

The `ping 8.8.8.8` command run successfully, and it means that the virtual machine now has the Internet access!

## Conclusion

Testo Framework allows you to connect your virtual machines to the Internet. This is done with the help of virtual networks, which could be also used to link up virtual machines with each other. Linking up machines with each other is the main highlight of one of the next guides.

You can find the complete test scripts for this guide [here](https://github.com/testo-lang/testo-tutorials/tree/master/hyperv/03%20-%20nat).
