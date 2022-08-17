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

This test scenario can be automated. Hypervisor API can be used to create virtual machines and to control keyboard/mouse input. Artificial neural networks can be used to detect whether some text is represented on the screen of the virtual machine and if so, where exactly it is located. If we combine a hypervisor API and nueral networks together, and add a simple language for writing down test scenarios we get Testo framework.

For example, the test scenario described above can be written as follows:

```
mouse click "Save"
type "Hello world"
wait "Complete"
```

## Framework overview

### Language for test scenarios

### Interpreter

### Nueral networks server

### Guest additions

## Usage samples

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
