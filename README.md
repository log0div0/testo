# Simple automation for complex tests

This is an official repository of the E2E-tests automatization framework called Testo. 

## Motivation

If you’ve tried to develop a more or less complex software product, you should’ve encountered the situations when, for some reason, it is impossible to automate the End-to-End (E2E) tests for your software. These reasons are many, let’s name a couple of them:

- The software doesn’t have (or just can’t have, for security reasons) an API you can hook up with.
- The software is legacy and was developed back in times when no one bothered with the tests’ automation.
- The software’s testing involves some third-party software (antivirus, for instance).
- The software must be tested under numerous target operating systems.
- You can’t test the software without a complex heterogeneous test bench with intermediate network nodes.

All these and many other obstacles lead to the worst nightmare of any developer — manual testing. But the worst part of it is that you can’t just test the software one time and leave it be. No, before each release (maybe even more often) you have to deploy your virtual machines, upload another software build, and do the same testing routine again and again, checking for possible regressions.

This is exactly the problem that Testo is trying to solve.

## What is Testo?

You see, a lot of E2E testing actually happens inside virtual machines. And any test run on a virtual machine can be automated with the sequence of simple actions, such as mouse movements and pressings of the keyboard buttons. These are the exact same actions that a QA engineer performs when manual testing a software product. Such tests could be represented, roughly speaking, as such:

1) Click the “Save” text on the screen.
2) Type the “Hello world” text on the keyboard.
3) Wait for the “Complete” text to appear on the screen.

And it doesn’t matter whether you’re testing a XAML-app, Qt-app, Electron-app, a web page or even a console application. You just click on the virtual machine’s screen and you don’t really care about the app’s internal design. Sounds convenient? Sure!

There’s only one catch: it’s not so easy to understand where the “Save” button is located on the screen, or whether the “Complete” text is present on the screen. I assume that’s one of the possible reasons why we don’t see the abundance of testing tools based on such concept.

On the other hand, the computer vision technology has made great steps forward recently, thanks to machine learning. Artificial neural networks (NN) handle even such difficult tasks as, for instance, driving cars. Surely, they can handle the much easier task of detection GUI objects on the screen, now can’t they?

As you could’ve guessed, the answer is yes. And you can see it for yourself when using Testo — a new system tests automation framework. Testo is essentially an interpreter, allowing you to run test scripts written in specially designed Testo-lang language. The scripts look somewhat like this:

```
mouse click "Save"
type "Hello world"
wait "Complete"
```

That’s all you need to write in Testo-lang to implement the scenario above!

However, I don’t want you to get the impression that Testo is just another Autoit or Sikuli look-alike. No, it’s not just any automation tool — it’s a whole framework designed for system tests automation. Testo takes care of numerous subtasks that a QA engineer may encounter: checking which tests should be re-run, the virtual test bench deployment, keeping track and reporting which tests failed or passed (and how exactly) and so on.


