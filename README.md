# Simple automation for complex tests

This is an official repository of the E2E-tests automatization framework called Testo. 

## Motivation

If you’ve tried to develop a more or less complex software product, you should’ve encountered the situations when, for some reason, it is impossible to automate the End-to-End (E2E) tests for your software. These reasons are many, let’s name a couple of them:

- The software doesn’t have (or just can’t have, for security reasons) an API you can hook up with.
- The software is legacy and was developed back in times when no one bothered with the tests’ automation.
- The software’s testing involves some third-party software (antivirus, for instance).
- The software must be tested under numerous target operating systems.
- The software requires multiple virtual machines at the same time for testing.

All these and many other obstacles lead to the worst nightmare of any developer — manual testing. But the worst part of it is that you can’t just test the software one time and leave it be. No, before each release (maybe even more often) you have to deploy your virtual machines, upload another software build, and do the same testing routine again and again, checking for possible regressions.

This is exactly the problem that Testo is trying to solve.

## What is Testo?

You see, a lot of E2E testing actually happens inside virtual machines. And any test executed on a virtual machine can be represented as a sequence of simple actions, for example:

1) Click the “Save” text on the screen.
2) Type the “Hello world” text on the keyboard.
3) Wait for the “Complete” text to appear on the screen.

And it doesn’t matter whether you’re testing a XAML-app, Qt-app, Electron-app, a web page or even a console application. You just click on the virtual machine’s screen and you don’t really care about the app’s internal design.

As you can guess this test scenario can be automated. Hypervisor API can be used to create virtual machines and to control keyboard/mouse input. Artificial neural networks can be used to detect whether some text (or UI element) is represented on the screen of the virtual machine and if so, where exactly it is located. If we combine a hypervisor API and nueral networks together, and add a simple language for writing down test scenarios we get Testo framework.

For example, the test scenario described above can be written as follows:

```
mouse click "Save"
type "Hello world"
wait "Complete"
```

## Framework overview

### Language for test scenarios

A special language was developed for compact recording of test scenarios. We call it Testo-lang. I'm not going to go into details here because there is an extensive documentation for that, but I'll say a few words about Testo-lang so that you have an idea of what it looks like.

As it was said, a test scenario is essentially a sequence of simple actions under a virtual machine. You don't need to create virtual machines manually. Instead you declare them as part of a test scenario:

```
machine my_super_vm {
    ram: 2Gb
    cpus: 2
    iso: "ubuntu_server.iso"
    disk main: {
        size: 5Gb
    }
}
```

This snippet instructs Testo-lang interpreter to create a virtual machine with 2Gb RAM, 2 CPU cores and 5Gb of disk space. The ISO “ubuntu_server.iso” is inserted in the virtual DVD-drive of the virtual machine, so when the machine is powered on, the Ubuntu Server installation pops up.

As soon as you have at least one virtual machine you can start writing the actual tests:

```
test my_super_test {
    my_super_vm {
        start
        wait "Language"
        press Enter
        wait "Install Ubuntu Server"
        press Enter
        # And so on
        ...
    }
}
```

This snippet declares a single test called `my_super_test` which uses only one virtual machine - `my_super_vm`. When running this test, the virual machine will be turned on first (`start` action). After that Testo-lang interpreter will wait for the "Language" text to appear on the screen of the virtual machine. If the text does not appear in a reasonable time - the test will fail. Then Testo-lang interpreter will press "Enter" key on virtual machine's keyboard. And so on.

This may seem counter-intuitive at first, because we consider the OS installation is just yet another test, on a line with any other regular software-checking tests. But it gets more reasonable if you imagine that you might develop the actual OS itself! Maybe you’re developing some custom OS (another Linux-based distribution, for example), or it’s just a simple just-for-fun toy OS. In any case, we do not make any distinction between testing OS and an application running inside of OS. The whole virtual machine is a system under test (SUT). That greatly simplifies testing application that actively interacting with OS or consist of several executable files.  

Looking ahead, I'll say that you don't have to install OS as a part of a test scenario. You can use a pre-installed VM image.

In fact, Testo-lang is heavily inspired by another language called CMake. Indeed, running tests with Testo-lang is very much like building a program from source codes. If you have ever compiled a program using CMake or simular build system, then you know that the program is rebuilt only if its source codes have been changes since the last build. Similar mechanism of "incremental test running" takes place in Testo-lang as well: the test is running only if its scenario or its dependencies have been changed. That's one of the main reasons why we decided to make our own language at all. 

### Interpreter

### Nueral networks server

### Guest additions

### Reporting tools

### Syntax highlighting

## Features

## Downloads

## Installation

### Debian/Ubuntu

### CentOS

### Windows

## Documentation

## Building from source

## Credits

## License
