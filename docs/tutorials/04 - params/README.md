# Tutorial 4. Params

## What you're going to learn

In this guide we will focus our attention on the param feature of Testo-lang.

## Introduction

During the previous guides we've written quite a lot of code for our test scripts, and it's not hard to see that there are several string constants that occur several times in the scripts. For example, `my-ubuntu-login` can be seen multiple times (during Ubuntu installation, then at the login attempts). If we'd decided to change the login to some other value we'd need to search all the file and replace the value several times.

We can also note that the paths to the Ubuntu Server and Guest Additions are absolute, which is not very convenient. If the ISO-images were located elsewhere, we'd need to adjust the test script. All that is certainly brings certain mess to the test scripts.

To solve this problem Testo-lang has the [params](../../reference/Params.md) mechanism. Params basically are global string constants. Let's take a look at them.

## What to begin with?

Let's remember how our script looks at the moment:

<Snippet id="snippet1"/>

We can see a few strings that occur several time during the script: `my-ubuntu` (hostname), `my-ubuntu-login` (login) and `1111` (password). Obviously, the more elaborate our script gets, the more such repeated constants will take place - and the easier it will be to make a mistake. So let's try to avoid that and declare three params:

```testo
param hostname "my-ubuntu"
param login "my-ubuntu-login"
param password "1111"

test ubuntu_installation {
	...
```

> Param declarations must be placed at the same level as virtual machines and tests declarations (e.g. globally). You can't declare params inside tests or other virtual entities declarations.

> It is prohibited to re-declare params.

Now inside our tests we can reference the params we've just declared:

```testo
...
param hostname "my-ubuntu"
param login "my-ubuntu-login"
param password "1111"

test ubuntu_installation {
	my_ubuntu {
		start
		...
		wait "Hostname:" timeout 30s; press Backspace*36; type "${hostname}"; press Enter
		wait "Full name for the new user"; type "${login}"; press Enter
		wait "Username for your account"; press Enter
		wait "Choose a password for the new user"; type "${password}"; press Enter
		wait "Re-enter password to verify"; type "${password}"; press Enter
		...
		unplug dvd; press Enter
		wait "login:" timeout 2m; type "${login}"; press Enter
		wait "Password:"; type "${password}"; press Enter
		wait "Welcome to Ubuntu"
	}
}

test guest_additions_installation: ubuntu_installation {
	my_ubuntu {
		plug dvd "/opt/iso/testo-guest-additions.iso"

		type "sudo su"; press Enter;
		# Take a note that you may reference params in any part of the string
		wait "password for ${login}"; type "${password}"; press Enter
		wait "root@${hostname}"
		...
	}
}

...
```

Now our test script looks much neater and easier to read. Additionally, if we'd decided to change our login, hostname or password we'd need to change the value in just one place.

## Passing params as command line arguments

And still there is one not-so-pretty thing in our test script: we have to specify full absolute paths to the ISO-images in the virtual machine declaration and in the `plug dvd` action when installing guest additions.

Of course, we could have declared a param `iso_dir "/opt/iso"`, but that would not fix the problem: our script would have still be bound to the exact iso-images placement. Obviously, this is not a good thing.

But, fortunately, in Testo-lang you can not only specify params statically inside test scripts, but pass them as command-line arguments as well. Let's try that:

```testo
machine my_ubuntu {
	cpus: 1
	ram: 512Mb
	disk main: {
		size: 5Gb
	}
	iso: "${ISO_DIR}/ubuntu_server.iso"
}
...
test guest_additions_installation: ubuntu_installation {
	my_ubuntu {
		plug dvd "${ISO_DIR}/testo-guest-additions.iso"
		...
	}
}
```

We changed our test script in such a manner that the paths to the Ubuntu Server and Guest Additions iso-images now contain a reference to the `ISO_DIR` param (you could also take a note that params are referencable inside the virtual machine declarations too). But we haven't declared the `ISO_DIR` param anywhere. If we tried to run the test script now, the same way we're used to, we would see an error:

<Asset id="terminal1"/>

Since the `ISO_DIR` param hasn't been declared, Testo can't resolve the reference to it and generates an error. So we'll try to pass the param `ISO_DIR` through a command line argument:

<Asset id="terminal2"/>

If the iso-images location changes for some reason (for example, the script is run on another computer), all we'll have to do is to change one command-line argument value when running the script, the script itself doesn't need to be modified.

If you've completed the guide 3 and run the script the way it was at the end of the previous guide, then now (after the new run) you'll see the next output:

<Asset id="terminal3"/>

Which means no tests were run. The reason is that all the tests are **cached** now, and there is no need to run them again. We will focus on caching mechanism in Testo-lang in the next guide, in which we'll provide a detailed explanation about how the tests managed to remain cached, despite the seemingly large changes in them.

But right now to make sure that out test script is still funcitonal, we need to run the interpeter with thw new command-line argument `--invalidate`, which resets the cache of the specified tests.

<Asset id="terminal4"/>

## Conclusions

You can make your test scripts more flexible and neater with params.

First, you can "rename" frequently used string constants so you can navigate through them easily.

Second, you can control the test scripts run with the command-line arguments, not changing the test scripts themselves.
