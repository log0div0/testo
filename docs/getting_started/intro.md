# Introduction

This documentation provides the information about generic concepts, getting started instructions and basic knowledge of Testo - system tests automation framework.

## Testo Framework usage motivation

Every software needs testing. Usualy testing is layered as such:

- Unit testing
- API testing
- Integration testing
- System testing

Unit testing can help validate source code works ocrrectly. It's targeted to individual functions and modules.

API testing (if there is an API to be tested) provides with validation of API calls in various situations.

Integration testing is a kind of complex testing when individual modules are combined and tested as a whole piece. It could also imply the testing of the whole program. The most important point in Integration testing is the abstraction from the software surroundings.

System testing (which is also called End-To-End testing) takes into account the whole software **and its surroundings**, including but not limited to: Operating System version, Network interaction (over a LAN or the Internet), specific hardware/driver combinations and so on.

System testing should be considered as one of the most complex kind of testing, with Systems Under Test (SUT) presented in the state they will be seen by the end users. The only possible unified way to automate this kind of tests is to automate the end user behaviour (e.g. human behaviour). System tests are the most difficult to automate.

Testo, which is a system test automation framework, addresses this problem and provides you with the opportunity to automate system tests for any kind of software (standalone software, a set of interacting programs or the whole operating system) by deploying and interacting with virtual machines. The testing is done via mimcking a real human working with actual virtual machines.

Essentially, human-machine interaction could be represented as a series of input actions (pressing buttons in a specific order, pressing power buttons, plugging/unplugging network links, flash drives and CDs/DVDs) and the output (e.g. screen contents) information analysis. If the expected reaction to an input differs from the actual picture on the screen, then the user (human) treats such a situation as an error. For example, a person expects to see the "Success" message, but can't find it on the screen for a quite long time.

This is exactly the kind of interaction that Testo Framework automates for you.
