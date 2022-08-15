# Guide 13. Importing disk images

## What you're going to learn

In this guide you're going to learn how to "import" manually prepared virtual machines into your test scripts.

## Preconditions

1. Testo Framework is installed.
2. Hyper-V is installed.
3. [Ubuntu Desktop 18.04](https://releases.ubuntu.com/18.04.4/ubuntu-18.04.4-desktop-amd64.iso) image is downloaded.
4. (Recommended) Testo-lang [syntax highlight](/en/docs/getting_started/getting_started#setting-up-testo-lang-syntax-highlighting) for Sublime Text 3 is set up.
5. (Recommended) [Guide 12](12_mouse) is complete.

## Introduction

In all our previous guides we've sticked to the concept of deploying the test benches from scratch. The virtual machines we created were absolutely blank: just a bunch of virtualized hardware with nothing installed. We had to automate everything: OS and guest additions installation, the network setups and so on. This approach has some great benefits to it:
1. All you need to deploy the test bench is a set of ISO-images and some small additional files you can keep in a VCS. So you can easily move the test bench to a new computer: just download the repository and the iso-images.
2. The test scripts are basically a documentation for the deploying process of the test bench: just read the scripts and you'll know what exactly happens to the test bench. From the very beginning!
3. You can easily adjust the test bench preparation steps. Just change any actions you want (in any part of the preparation process) and Testo will do the rest to get the test bench into the state you want it to be.

However, the approach has some downsides as well:
1. Sometimes the preparations are just too large and tedious to automate, and, therefore, developing the test scripts from scratch is just too long and ineffective.
2. Sometimes you want to just focus on the actual tests for the SUT and you don't want to spend your time automating the secondary virtual machines setups. Maybe it would be so much easier for you to just do some manual job preparing your secondary virtual machines and then just import the results into the test scripts.

So the Testo Framework gives you the opportunity to take a different approach for deploying the test becnhes: you may prepare a virtual machine yourself (or just get it from somewhere) and then import its disk into a virtual machine in your test scripts. This way you can start the scripts not from scratch, but from the point you see as the most preferable. And this is what we're going to learn about in this guide.

## What to begin with?

Let's take, for instance, the Ubuntu Desktop 18.04 installation. In the previous guide we've managed to automate the installation for this OS using mostly the mouse. But let's assume that we don't want to automate the OS installation. That we just want to do it manually one time, then stage our efforts and import the result as the starting point for the `ubuntu_desktop` machine.

Well, let's get right to it.

For starters, open Hyper-V manager and create a new virtual machine (we're going to call it `handmade`). The configuration may be arbitrary (just don't forget to specify the Ubuntu Desktop iso image), with the exception for the main disk size:

This is the disk we'll later import into the `ubuntu_desktop` machine, so let's give it the size of 10 Gigabytes (just like in the previous guide).

When the virtual machine is created, just install the OS manually.

After the virtual machine is installed, let's do a super-quick "setup" for it. To mimick a setup let's create an empty folder `Handmade_folder` on the Desktop, so later on we could make sure that we actually imported the `handmade` machine successfully.

![Handmade folder](/static/docs/tutorials/hyperv/13_disks_import/Handmade_folder.png)

Now turn off the `handmade` machine, since we're ready to import the results into the test scripts.

## Importing the `handmade` machine disk into the test scripts

Now with the manual setup finished, we need to move back to the script developing. Let's create a new `.testo` file, name it `handmade.testo` and declare a virtual machine `ubuntu_desktop` like this:

```testo
machine ubuntu_desktop {
	cpus: 1
	ram: 2Gb
	disk main: {
		source: "${VM_DISK_POOL_DIR}\\handmade.vhdx"
	}
	nic internet: {
		attached_to: "internet"
	}
}

test check_handmane_ubuntu {
	ubuntu_desktop {
		start
		wait "Handmade" timeout 3m
	}
}
```

There are two interesting points in the virtual machine declaration:
1. The `main` disk now has the `source` subattribute instead of the `size` subattribute. In the `source` subattribute we need to specify the path to the `handmade.vhdx` disk image. When the `ubuntu_desktop` is created, this image will be copied and imported into its configuration. The original `handmade.vhdx` isn't going to be affected in any way.
2. `ubuntu_desktop` doesn't have the `iso` attribute. Indeed, there is no need for the installation medium now, since the Ubuntu is already installed.

We also developed a very basic `chack_handmade_ubuntu` test, in which we're going to make sure that the import was a success. The virtual machine is created in the turned off state (as usual), so first we need to start it. Then we're waiting for the Desktop to appear with the `Handmade` folder on it. Once the folder is detected we'll know that everything is OK.

> There is no login prompt because we chose "Log in automatically" when installing Ubuntu manually. If you'd chosen "Require my password to log in" - then please adjust the `check_handmade_ubuntu` test accordingly.

Let's run the script (take a look at the `VM_DISK_POOL_DIR` param value - this is the default path to the disk images of the virtual machines created with virt-manager).

<Asset id="terminal1"/>

We can see that the test passed successfully, which means, that the `Handmade_folder` was detected after all. The manually-created virtual machine import is complete!

## Conclusions

There are two approaches to virtual bench deploying in Testo Framework: doing everything from scratch and importing existing virtual machines' disks into the test scripts. Importing existing disks may be convenient and useful if you don't want to spend your time automating preparatory tests. "Imported" virtual machines have the same caching rules as regular ones.

You can find the complete test scripts [here](https://github.com/testo-lang/testo-tutorials/tree/master/hyperv/13%20-%20disks%20import).
