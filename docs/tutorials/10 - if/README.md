# Tutorial 10. If clause

## What you're going to learn

In this guide you're going to learn about:
1. If-statements in Testo-lang.
2. Complex select expressions in the `wait` actions.

## Introduction

In the previous guide we managed to improve our code so it would be cleaner and simpler to navigate through. However, there is at least one more moment that can be improved.

You may remember several occasions when we had to adjust the Ubuntu Server installation test. We had to do it because the installation process depends on the virtual machine configuration: does the machine have any NICs, does it have the Internet access and so on. At the moment the `install_ubuntu` macro works fine only in the following conditions:

1. The virtual machine has two or more NICs;
2. The virtual machine has the Internet access;
3. The root password is weak enough for the corresponding warning to appear.

But what if we want our macro to work in any conditions? What if we want our macro to install the Ubuntu Server successfully no matter what the virtual machine configuration is?

Obviously, the macro needs to apply the actions a bit differently, depending on the current situation. And there is a tool in the Testo-lang just for that - the [`if-else` statements](../../reference/Conditions.md). You can control the action flow based on string constants, params' values and the actual screen contents. In this guide we're going to try out both simple if-expressions and more complex ones, with screen contents checks.

## What to begin with?

Let's begin with adjusting the `install_ubuntu` macro so it would work with both weak and strong admin passwords. To do that, let's figure out the difference with the action flow in both cases.

Right now, the script is OK only for weak passwords and look like this:

```testo
macro install_ubuntu(hostname, login, password = "${default_password}") {
	start
	...
	wait "Choose a password for the new user"; type "${password}"; press Enter
	wait "Re-enter password to verify"; type "${password}"; press Enter
	wait "Use weak password?"; press Left, Enter
	wait "Encrypt your home directory?"; press Enter
	...
}
```

The important part is that we expect the warning `"Use weak password?"` to appear on the screen, after which we need to press Left and Enter.

Obviously, when using a strong password, the warning won't show up, and the macro wouldn't need to wait for the warning to appear:

```testo
macro install_ubuntu(hostname, login, password = "${default_password}") {
	start
	...
	wait "Choose a password for the new user"; type "${password}"; press Enter
	wait "Re-enter password to verify"; type "${password}"; press Enter
	wait "Encrypt your home directory?"; press Enter
	...
}
```

So we're going to combine both possible cases inside a single `if` clause.

> Keep in mind that for now we're not considering the best possible solution for the problem. The best solution (checking the screen contents) is considered a bit later in this guide. The current proposal is for educational puproses.

```testo
macro install_ubuntu(hostname, login, password="${default_password}", is_weak_password="") {
	start
	...
	wait "Choose a password for the new user"; type "${password}"; press Enter
	wait "Re-enter password to verify"; type "${password}"; press Enter
	if ("${is_weak_password}") {
		wait "Use weak password?"; press Left, Enter
	}
	wait "Encrypt your home directory?"; press Enter
	...
}
```

We've added a new macro argument - `is_weak_password` - to control the macro action flow. This argument works as a hint for the macro and tells it whether the password is weak or not. If the argument length is zero, then the `if`-expression returns `true`, and the corresponding action block (with waiting for the warning) is executed. If the length is greater than zero, the expression returns `false` and the action branch is skipped.

We developed the macro in such a way, that the `${default_password}` needs to be strong enough for the macro to work. The current `${default_password}` value (`1111`) is not suitable, so the macro wouldn't work if called with the default value arguments. Let's see for ourselves.

<Asset id="terminal1"/>

As we'd expected, the test didn't go as planned at all. Therefore, we need to adjust the `default_password` value to make it a strong enough password:

```testo
param default_password "ThisIsStrongPassword"
```

Run the script again. Keep in mind, that we run **only** the base test `sever_install_ubuntu`:

<Asset id="terminal2"/>

So the Ubuntu Installation test is OK once again. Now let's assume, that for some reason we want to set the weak password on the `client` machine. To do that, we just need to modify the macro call in the `client_install_ubuntu` test like this:

```testo
test client_install_ubuntu {
	client install_ubuntu("${client_hostname}", "${client_login}", "1111", "yes")
}
```

In this call we explicitly pass the `1111` password instead of the `default_password`. To give the macro a hint that the password is, indeed, weak, we pass a non-zero-length string as the fourth argument.

<Asset id="terminal3"/>

We can see, that our solution works as planned, and now the `install_ubuntu` macro works equally good both with weak and strong passwords. But, of course, controlling the action flow this way is not very convenient. If our macro had more if-branches (and that's going to happen pretty soon), we would have to add more "hint" arguments, and in the end their number would be so high that the macro would have been simply unusable.

But, luckily, in Testo-lang you can check the screen contents in the `if` statements.

## Checking the screen contents in the `if` expressions

You can check the screen contents in the if-statements with the `check` expressions. The syntax for the `check` expressions is very similar to the `wait` actions, but instead of generating an error if the specified event doesn't show up on the screen before the timeout, the `check` expression returns `false` in the same situations (and `true` if the event does show up). The `check` expessions are to be used only inside the `if` statements.

Let's see a `check` expression in action:

```testo
macro install_ubuntu(hostname, login, password = "${default_password}") {
	..
	wait "Re-enter password to verify"; type "${password}"; press Enter
	if (check "Use weak password" timeout 3s) {
		press Left, Enter
	}
	wait "Encrypt your home directory?"; press Enter
	...
```

Instead of using artificial hints with additional arguments, now the macro depends on the actual state of the virtual machine screen. Our `if`-statement means this: if the `Use weak password` text appears on the screen in 3 seconds, the action `press Left, Enter` must be applied. Otherwise no additional actions are needed. If the text appears less than in 3 seconds, the check is triggered immediately.

If we hadn't specified `timeout 3s`, then by default the `check` expression would've checked the screen contents **just** one time and returned the result immediately. But that's not exactly what we want, because the `Use weak password` screen doesn't appear immediately after the previous screen. It could take, let's say, 0,5-1 second. The way we used the `check`, basically means this: "Well, if the `Use weak password` screen doesn't appear in 3 seconds, it won't appear at all".

Naturally, we don't need the `is_weak_password` argument anymore and we can remove it without any sore feelings.

Don't forget to adjust the macro call in `client_install_ubuntu` and run the scripts once again.

```testo
test client_install_ubuntu {
	client install_ubuntu("${client_hostname}", "${client_login}", "1111")
}
```

<Asset id="terminal4"/>

Both tests passed successfully, so now it's time to move on.

## One macro - various NICs numbers

We've taken care of handling the situations with both weak and strong passwords. But there is another problem we've encountered several times: with various NICs number, the installation process varies as well.

Let's take a closer look at this piece of script of the Ubuntu installation:

```testo
wait "Keyboard layout"; press Enter
#wait "No network interfaces detected" timeout 5m; press Enter
wait "Primary network interface"; press Enter
wait "Hostname:" timeout 5m; press Backspace*36; type "${hostname}"; press Enter
```

There are 3 possible situations:

1. If there is no NIC at all, the "No interfaces found" warning appears on the screen, in which case we need to perform an additional `press Enter` action. After that, the `Hostname` screen shows up.
2. If there is exactly one NIC, then the `Hostname` screen appears, right after the `Keyboard layout` screen, and no additional warnings show up.
3. If there are two or more NICs, then the primary interface screen selection shows up. After the selection, the `Hostname` screen appears.

Turns out, we can't be sure beforehand what screen is going to appear after the `keyboard layout`: either `No network interfaces detected` or `Primary network interface` or `Hostname`. So how do we squeeze all three possibilities in one macro?

To achieve that, we're going to do something which resembles a `switch-case` clause in other languages, but looks a bit different.

For starters, we make use of the `wait` feature to wait not a single textline, but whole seleciton expressions (see [here](../../reference/Actions.md#select-expressions-for-the-wait-and-check-actions) for more information):

```testo
wait "Keyboard layout"; press Enter
wait "No network interfaces detected" || "Primary network interface" || "Hostname:" timeout 5m
```

> Complex selection expressions can be used both in `wait` actions and `check` expressions.

With this `wait`, we establish that we're OK with at least one of the specified textlines to appear on the screen in 5 minutes.

We're halfway through already.

But just waiting the screen is not enough. We also have to apply different actions based on the exact screen we've got. And that's where the already known `if(check)` statements come to action:

```testo
wait "No network interfaces detected" || "Primary network interface" || "Hostname:" timeout 5m
if (check "No network interfaces detected") {
	press Right, Enter
} else if (check "Primary network interface"){
	press Enter
}
wait "Hostname:" timeout 5m; press Backspace*36; type "${hostname}"; press Enter
```

So basically, we get the following picture: first we wait for any of the expected screens to appear, and then we try to find out which screen exactly appeared. Then we apply the actions accordingly to the situation. You may think that we're missing the `else` section for the `Hostname` case, but since the `Hostname` screen appears in the end anyway, we can spare ourselves a few lines of code.

> Take a note, that in this case the `check` expressions use the default `timeout`, which means that we're interested in the instant screen contents, not in a period of time. We can do this because we're sure that the screen already showed up, and we don't need to wait anything.

Ok, moving on. The next moment waiting for our inspection lies in the next macro piece:

```testo
wait "Encrypt your home directory?"; press Enter
#wait "Select your timezone" timeout 2m; press Enter
wait "Is this time zone correct?" timeout 2m; press Enter
wait "Partitioning method"; press Enter
```

Depending on whether the virtual machine has the Internet access or not, different screens appear: either with the manual timezone selection or with a confirmation for the guessed timezone. Fortunately, in both cases we need to do the same action: `press Enter`. So we can generalize this piece without any if-statements:

```testo
wait "Encrypt your home directory?"; press Enter
wait "Select your time zone" || "Is this time zone correct?" timeout 2m; press Enter
wait "Partitioning method"; press Enter
```

Done! Now the `install_ubuntu` macro works with different virtual machine configurations and with both weak and strong passwords. We can move this macro in the standalone file and forget about it. Now to install Ubuntu Server in different circumstances we can just call the macro, without worrying about its implementation.

## Putting HTTP_PROXY into action

Another moment worth consideration - the need to enter the HTTP_PROXY value in the case your virtual machine is placed behind a proxy-server. Up until this moment we've been assuming that the Host has the direct Internet access, but if we really want to prepare the Ubuntu installation macro for all possible situations, we have to remember about the HTTP_PROXY.

Naturally, the presense of the proxy server should be an externally-defined param (like the `ISO_DIR` param) and to pass this information we would need to specify the `--param HTTP_PROXY "192.168.1.1"` command line argument.

The question is: how to make the macro work with both the `HTTP_PROXY` defined and not defined? We can do it like this:

```testo
...
wait "HTTP proxy information" timeout 3m;

if (DEFINED HTTP_PROXY) {
	type "${HTTP_PROXY}"
}

press Enter
...
```

The `DEFINED` expression checks whether the `HTTP_PROXY` param is defined or not. Keep in mind that the param name is specified as an identifier, not a string.

If the `HTTP_PROXY` is defined, we need to type its value and then press Enter.

> Keep in mind, that if we'd tried to apply the `type "${HTTP_PROXY}"` action with the `HTTP_PROXY` param not deinfed, we would've gotten an error (referencing undefined param is prohibited).

## Conclusions

With if-statements you can control the action flow depending on various circumstances (including the different screen contents). With this tool you can develop more flexible and generalized macros and tests.

> Of course, we haven't learned the whole possibilities of the if-statements in this guide. In the if-statements you can use the whole selection expressions, including unary and binary operators, comparisons and so on. For more information see the [documentation](../../reference/Conditions.md).
