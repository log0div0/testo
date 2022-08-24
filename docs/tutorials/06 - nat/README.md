# Guide 6. Internet access in virtual machines

## What you're going to learn

In this guide you're going to:

1. Learn about virtual networks and Network Interface Cards (NICs).
2. Learn how to provide the Internet access to virtual machines inside your test scripts.

## Preconditions

1. Testo Framework is installed.
2. Virt manager is installed.
3. The Host has the Internet access.
4. [Ubuntu server 16.04](https://releases.ubuntu.com/16.04/ubuntu-16.04.7-server-amd64.iso) image is downloaded and located here: `/opt/iso/ubuntu_server.iso`. The location may be different, but in this case the `ISO_DIR` command-line param has to be adjusted accordingly.
5. Testo guest additions iso image is downloaded and located in the same folder as Ubuntu Server 16.04 iso-image.
6. (Recommended) Testo-lang [syntax highlight](/en/docs/getting_started/getting_started#setting-up-testo-lang-syntax-highlighting) for Sublime Text 3 is set up.
7. (Recommended) [Guide 5](05_caching) is complete.

## Introduction

Aside from virtual machines, there are two other virtual entities available in Testo-lang: virtual flash drives and virtual networks. Virtual flash drives are described in the future guides; right now we're going to focus on the virtual networks.

Generally, virtual networks in Testo-lang could be used for 2 purposes: to link up virtual machines with each other and to provide the Internet access to a virtual machine. Linking up machines is explained in the next guide, so in this guide we're going to learn about the Internet access.

## What to begin with?

At the moment we have a set of test scripts with the Ubuntu OS and guest additions installations on the virtual machine `my_ubuntu`. Usually, there is little use in a "bare" Ubuntu Server and most of the times you want to install some additional software on your server. Yes, theoretically you could've prepared all the necessary software beforehand and copy it inside the guest via `copyto`, but it is so much easier to use the usual Ubuntu package repository. Obviously, you'd need the Internet access for this to happen.

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
	iso: "${ISO_DIR}/ubuntu_server.iso"

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

<Asset id="terminal1"/>

If your tests are cached at the moment, you'll see that the `ubuntu_installation` test has lost its cache and needs to be re-run. It could seem strange, since we haven't modified the test itself, and its cache should've stayed intact. Hovewer, in the last guide we've mentioned that the cache validity depends on the involved virtual machines configurations integrity. Since we've changed the `my_ubuntu` configuration, all the tests referencing this virtual machine must be run again.

In any case, after a few minutes you're going to find out that the `ubuntu_installation` test now fails. The output gives us a hint, that Testo couldn't detect the "No network interfaces detected" text in 5 minutes.

Indeed, if you open virtual manager and check out the virtual machine screen, you'll see this:

![Hostname](/static/docs/tutorials/qemu/06_nat/hostname.png)

The reason for this screen to appear is that we added the NIC to `my_ubuntu`, so now, naturally, the warning about not detecting any interfaces doesn't show up.

Let's comment up the line with waiting for this warning. But now 30 seconds may be not enough for the next text "Hostname" to appear, so let's enlarge this timeout to 5 minutes:

```testo
...
wait "Country of origin for the keyboard"; press Enter
wait "Keyboard layout"; press Enter
#wait "No network interfaces detected" timeout 5m; press Enter
wait "Hostname:" timeout 5m; press Backspace*36; type "${hostname}"; press Enter
wait "Full name for the new user"; type "${login}"; press Enter
wait "Username for your account"; press Enter
...
```

And run the script again.

If you don't have any issues with the proxy server (or if you don't use it at all), then the test script might fail one more time.

<Asset id="terminal2"/>

Instead of the screen with the time zone selection you'll see another screen:

![Timezone](/static/docs/tutorials/qemu/06_nat/timezone.png)

The reason is that, thanks to the Internet access, Ubunstu Server installator now can detect your current timezone automatically. So let's change our script a little bit more:

```testo
...
wait "Re-enter password to verify"; type "${password}"; press Enter
wait "Use weak password?"; press Left, Enter
wait "Encrypt your home directory?"; press Enter

#wait "Select your timezone" timeout 2m; press Enter
wait "Is this time zone correct?" timeout 2m; press Enter
wait "Partitioning method"; press Enter
...
```

Now all the tests should pass successfully.

Let's check out that the Internet is indeed available inside our tests scripts. First, we rename the `guest_additions_demo` test into `check_internet` and modify it as the following:

```testo
test check_internet: guest_additions_installation {
	my_ubuntu {
		exec bash "apt update"
	}
}
```

Run the script and you'll see the next output:

<Asset id="terminal3"/>

In the `exec bash` action output we can clearly see the successful run of the `apt update` bash command. Which means that we are connected to the Internet!

## Conclusions

Testo Framework allows you to connect your virtual machines to the Internet. This is done with the help of virtual networks, which could be also used to link up virtual machines with each other. Linking up machines with each other is the main highlight of the next guide.

You can find the complete test scripts for this guide [here](https://github.com/testo-lang/testo-tutorials/tree/master/qemu/06%20-%20nat).
