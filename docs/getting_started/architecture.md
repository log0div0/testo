# Overview

## Main components

Test Framework is based on three main components:

1. Specially developed test script language named Testo-lang and the interpreter for this language.
2. Interaction with hypervisors. Testo calls hypervisors API while interpreting test scripts.
3. Neural networks-driven virtual machine screen contents detection. 

<img src="/static/docs/getting_started/general_en.svg"/>

### Testo-lang language

To run test scripts with Testo Framework you have to perform two steps: declare the virtual infrastructure and define the actions to be applied to that infrastructure. Testo-lang gives you a convenient way to declare virtual machines, flash drives and networks, the totally of which will determine the virtual test bench. Additionally, Testo-lang gives you the most intuitive way to describe actions to be applied to the virtual test bench. Those actions could be both mimicking human behaviour (type a text on the keyboard, move the mouse cursor, plug a flash drive, turn off the power and so on) and representing the more high-level commands, which are avaliable after an agent (testo-guest-additions) inside the virtual machine (execute a bash-script, copy a file from the host to the virtual machine and so on).

Such an approach allows Testo Framework to deploy test benches "on the fly", without any user-made pre-setups. All the Testo user has to do is to run test scripts, and the virtual test bench will be created and deployed automatically. This is similar to compiling a program from source code - all that is needed for compilation is nothing more than a bunch of source code files.

With the tests scripts written, the interpreter `testo` steps in. This interpreter interacts with the hypervisor, thus interpreting the tests.

### Interaction with hypervisors

To run the tests, Testo Framework converts the scripts in `.testo` files to the set of commands for the hypervisor, which manages the virtual infrastructure. At the moment Testo can work with the [QEMU](https://www.qemu.org/) hypervisor in Linux-based operating systems and with the [Hyper-V](https://docs.microsoft.com/en-us/windows-server/virtualization/hyper-v/hyper-v-technology-overview) hypervisor in Windows 10. Hyper-V integration is implemented in experimental mode: some feature are not available at the moment. 

### Neural Networks

The main validating action in Testo Framework is the `wait` action. This action allows you to determine whether some event did or did not occur on the virtual machine screen. For example, with the `wait` action you can determine whether some text or image has been displayed. If the text/image hasn't appeared on the screen for some time period, this action will cause the test to fail. For example, if a user starts a setup process for some prorgramm, he expects to see the text "Success" in the next 10 minutes tops. If the text is not here in 10 minutes, the user may be sure that something went wrong, and the test must be ended with an error.

Testo Framework uses Neural Networks for screen event detection. Testo takes screenshots from the virtual machine screen for some specified time interval. These screenshots are then processed in the Object Character Recognition Engine. If the text is found, then the `wait` action will complete successfully and the test will continue. The test will fail otherwise.

Since the search is done with only the Neural Network Engine, Testo Framework does not require the presense of any special agents inside the virtual machine and can equally easy process screen contents of FreeBSD console output, or Windows 10 GUI.

## Guest Additions

By default the Testo Framework allows you to apply actions mimicking human behaviour to virtual machines. This could be typing text on the keyboard, waiting for an event to occur on the screen, turn on/turn off the power and so on. This approach gives the Testo Framework the opportunity to treat the System Under Test (SUT) as a "black box", since there's no need for any additional agents precense (which may affect the tests outcome) inside the SUT.

<img src="/static/docs/getting_started/general-negotiator_en.svg"/>

On the other hand, sometimes it is not necessary to follow the "black box" approach. This can happen when, for example, the actions are applied to the auxiliary system, not the main System Under Test. In this case there's a way to simplify the interaction with virtual machines in Testo Framework. You may install the guest agent `testo-guest-additions`, which is distributed alongside the `testo` interpreter, inside the guest virtual machine. After this installation new high-level actions become avaliable: `exec bash` (execute a bash script), `copyto` (copy the files from the host to the guest) and so on.

Agent `testo-guest-additions` is a special system service with administrative privileges. Testo Framework calls this service directly while interpreting high-level actions, bypassing hypervisor API.

Guest Additions, if used, could make the test scripts much easier.
