# Guide 11. No snapshots

## What you're going to learn

In this guide you're going to learn about the tests without the hypervisor snapshots in Testo Framework. With this kind of tests you can save a lot of disk space.

## Preconditions

1. Testo Framework is installed.
2. Virt manager is installed.
3. Host has the Internet access.
4. [Ubuntu server 16.04](https://releases.ubuntu.com/16.04/ubuntu-16.04.7-server-amd64.iso) image is downloaded and located here: `/opt/iso/ubuntu_server.iso`. The location may be different, but in this case the `ISO_DIR` command-line param has to be adjusted accordingly.
5. Testo guest additions iso image is downloaded and located in the same folder as Ubuntu Server 16.04 iso-image.
6. (Recommended) Testo-lang [syntax highlight](/en/docs/getting_started/getting_started#setting-up-testo-lang-syntax-highlighting) for Sublime Text 3 is set up.
7. (Recommended) [Guide 10](10_if) is complete.

## Introduction

As you could've noticed, the tests caching plays a huge role in Testo Framework. It saves you a lot of time by using results of the already successfully run tests (if their cache is valid, of course), thus avoiding unnecessary test runs. This feature is possible thanks to the hypervisor ability to take and restore snapshots of virtual machines and flash drives.

But this approach has a downside as well: every snapshot takes a lot of disk space, and you could run out of this space pretty fast. The situation gets worse when you consider the fact that at the end of the test all virtual entities get their own snapshot. For example, if a test involves 5 virtual machines and 2 flash dirves, then you'll get 5 virtual machine snapshots and 2 flash drive snapshots.

And so, to save you some disk space, there is a feature in Testo-lang that gives you the opportunity to create tests without the **hypervisor** snapshots, with only light-weight metadata files. With this feature used properly you'll save a ton of disk space without any significant damage to the test runs, and that is the topic of the today' guide.

## What to begin with?

Let's take a look at the tests hierarchy we've got to this point:

<img src="/static/docs/tutorials/qemu/11_no_snapshots/test_hierarchy.svg"/>

We have 10 tests in total, at the end of each test snapshots are created. We already consume a huge amount of disk space as it is. Of course, we want to fix this issue.

Let's figure out why do we even need snapshots at the end of each successful test. Mostly - so the Testo can restore those snapshots of virtual machines and flash drives when it is necessary to run the children tests. For example, if the `test_ping` test had lost its cache, Testo Framework would've required the snapshots from the `server_prepare` and `client_prepare` tests just to run the requested test.

But to think about it, why do we even need the snapshots at the end of the `test_ping` and `exchange_files_with_flash` tests? These tests are the leaves in our tests tree and there's just no need to restore the virtual test bench state from the end of these tests. So, therefore, we may just tell Testo not to create hypervisor snapshots at the end of them (please make sure that all your tests are passed and cached before making the changes):

```testo
[no_snapshots: true]
test test_ping: client_prepare, server_prepare {
	client exec bash "ping 192.168.1.2 -c5"
	server exec bash "ping 192.168.1.1 -c5"
}

[no_snapshots: true]
test exchange_files_with_flash: client_prepare, server_prepare {
	client exec bash "echo \"Hello from client!\" > /tmp/copy_me_to_server.txt"
	copy_file_with_flash("client", "server", "exchange_flash", "/tmp/copy_me_to_server.txt", "/tmp/copy_me_to_server.txt")
	server exec bash "cat /tmp/copy_me_to_server.txt"
}
```

We just used a new Testo-lang feature: [tests attributes](/en/docs/lang/test). At the moment there're only two available test attributes: `no_snaphots` and `description`. The `description` attribute is not so much interesting - it allows you to create a human-readable test description, which may be stored in the tests report (if you tell Testo to create such a report with the `--report_folder` command-line argument). But the `no_snapshots` attribute is more meaningful, and we're going to set its value to `true`.

Let's run the script:

<Asset id="terminal1"/>

We can see that both of our modified tests had lost their cache and was run again. The reason is that test attributes are included in tests checksums.

But what's now? Now the hypervisor snapshots hadn't been created at the end of the test, so we could've assumed that the tests wouldn't going to be cached again, and they would be running all the time, right? Wrong! Let's run the tests again:

<Asset id="terminal2"/>

So what do we see? All the tests remained cached and nothing had been run! And that's with two of our tests missing the hypervisor snapshots (which you could see for yourself in the virtual manager):

![No snapshots](/static/docs/tutorials/qemu/11_no_snapshots/no_snapshots.png)

Why does this happen? Let's sort this out.

The thing is, there are two types of snapshots in Testo Framework. Both types work independently:

1. Metadata snapshots. These are essentially small text files created by Testo Framework at the end of each test. You can't do anything with them. The files contain the various information about the tests helping Testo validate the cache. If you take a real close look at the last terminal output we'd got when run the `no_snapshots` tests, you'd still see the `Taking snapshot...` message - this actually implies metadata snapshots.
2. Hypervisor snapshots. These are the snapshots we're all familiar with. This kind of snapshots are created only if there is no `no_snapshots` attribute specified for the test (or its value is `false`, which is the default value). Since we'd turned this attribute on, the hypervisor snapshots weren't created.

We can sum everything up with an important conclusion:

> The `no_snapshots` attribute doesn't affect the test caching. A test with this attrubute is cached like any other. The attribute **doesn't mean** that the test is going to be run every time.

Turns out, we've saved up a little disk space and lost absolutely nothing, since the `test_ping` and `exchange_files_with_flash` snapshots aren't of any use for us. This gives us another important conclusion:

> You can put the `no_snapshots` attribute in all the "leaf" tests (tests with no children) with literally no damage at all, since you're not going to restore your test bench into those states anyway.

## no_snapshots in the intermediate tests

You might've got the impression that if the `no_snapshots` saves up the disk space and doesn't affect the tests caching, then, maybe, it should be put into each and every test? That impression would've been wrong.

Yes, this attribute doesn't affect the caching, but it doesn't mean there is no negative side effects. Let's demonstrate these effects and add this attribute to the `client_unplug_nat` test:

```testo
[no_snapshots: true]
test client_unplug_nat: client_install_guest_additions {
	client unplug_nic("${client_hostname}", "${client_login}", "nat", "1111")
}
```

Now let's run this test and nothing more.

<Asset id="terminal3"/>

Let's also make sure that the test is cached, despite the `no_snapshots: true` attribute:

<Asset id="terminal4"/>

And now run the test `client_prepare`, which depends on the `client_unplug_nat` test:

<Asset id="terminal5"/>

We can see a very peculiar thing: the `client_unplug_nat` test is marked both as `UP-TO-DATE` and as `TEST TO RUN`. Let's sort this out.

When Testo Framework scans the tests tree trying to figure out which tests are supposed to be run and which are cached, each test is evaluated individually. Since we want to run the `client_prepare` test, then first all of its parents' cache is probed. This is done for `client_install_ubuntu`, `client_install_guest_additions` and `client_unplug_nat`. All these tests have the valid cache, so they are marked as `UP-TO-DATE`, which we can see in the output.

Then comes the time to check the cache for the `client_prepare` test itself. The cache is invalid (because we'd earlier changed the `client_unplug_nat` parent-test) and the test must be re-run. But how can we run it?

If the `client_unplug_nat` test hadn't been marked with the `no_snapshots` attribute, we could've restored the virtual machine states as they were at the end of the `client_unplug_nat` test. But this test doesn't have the hypervisor snapshots, so we have nowhere to restore the virtual machines into. This raises the question: "How to revert the `client` machine into the state it was at the end of the `client_unplug_nat` test?" Well, to do so, Testo Framework searches the tests tree trying to find a test with the hypervisor snapshots turned on, so it can play the part of the "starting point". In our case, the `client_install_guest_additions` test is going to be selected.

Testo restores the `client` machine into the `client_install_guest_additions` state and it begins to re-run the `client_unplug_nic` test **just** to restore the `client` machine into the `client_unplug_nic` state. And that's why we can see the `client_unplug_nic` in the `TESTS TO RUN` queue.

When the `client` machine is in the correct state, we can, finally, run the `client_prepare` test itself. The whole process may be visualized as this:

<img src="/static/docs/tutorials/qemu/11_no_snapshots/search_en.svg"/>

If the `client_install_guest_additions` also had the `no_snapshots` attribute, the resulting test plan to run the `client_prepare` test would've looked like this: `client_install_guest_additions->client_unplug_nat->client_prepare`.

And now let's try to run all the tests at once:

<Asset id="terminal6"/>

So what do we see? We can see that despite the `client_unplug_nat` test now has no hypervisor snapshots, the **leaf-tests** run as usual: because we still have the virtual machine snapshots from the `client_prepare` test.

> Turns out, the `no_snapshots` attribute may be good for disk space saving, but sometimes at the cost of increasing time of test runs.

Try to add the `no_snapshots` attribute to the `server_unplug_nat` and investigate which tests are going to run and when.

Now let's turn our attention to one more thing, after which we're going to state a few basic rules about setting the `no_snapshots` attibute.

## no_snapshots in "anchor" tests is a bad idea

Before proceeding further, make sure that the `client_unplug_nic`, `server_unplug_nic`, `test_ping` and `exchange_files_with_flash` tests have the `no_snapshots: true` attribute and have been cached up.

With things arranged this way, we've managed to save quite a lot of disk space, and the `test_ping` and `exchange_files_with_flash` tests run just as quickly as before, with the condition that we don't touch the `client_prepare` and `server_prepare` tests, so they won't lose their cache. We've reached a certain point of balance: we consume not so much disk space and we don't get a lot of inconveniences with the tests runs.

But let's demonstrate what's going to happen if we push the limit too far.

Let's add the `no_snapshots` attribute to the `client_prepare` and `server_prepare` tests and run everything:

<Asset id="terminal7"/>

Just look at how big the `TESTS TO RUN` queue had got! We can see that the `server_unplug_nat`, `client_unplug_nat`, `server_prepare` and `client_prepare` are scheduled to run two times each! Let's figure out what's happening:

1. We need to run two leaf tests: `test_ping` and `exchange_files_with_flash`, which depend on the parent-tests `client_prepare` and `server_prepare`.
2. Since the `client_prepare` and `server_prepare` tests don't have the hypervisor snapshots, Testo Framework is forced to find the closest tests with the hypervisor snapshots enabled.
3. For `test_ping`, the running path is organized like this: `server_unplug_nat->server_prepare->client_unplug_nat->client_prepare->test_ping`.
4. The same path is formed for the `exchange_files_with_flash` test as well! That is the only way to restore the virtual machines state so that the leaf could to be run. And therefore, some tests are scheduled to run 2 times.

Yes, we saved some more disk space, but at what cost? The tests running time increased vastly: the disadvantages are significantly greater than the benefits.

So the question is raised: are there some general rules about which tests should get the `no_snapshots` attribute and which tests shouldn't? I suggest the following rules:

1. All the leaf tests (tests without any children) should get the `no_snapshots` attribute, since there is no damage hidden there.
2. The intermediate tests should get the `no_snapshots` attribute if they are not **anchor** tests. A test is considered an anchor, if its results are often restored when running its children tests.
3. Tests with multiple children **should not** get the `no_snapshots` attribute.

If we apply these rules to our tests tree, we will get this:

1. `test_ping` and `exchange_files_with_flash` are leaf tests, so they should get the `no_snapshots` attribute.
2. `client_prepare` and `server_prepare` definetely shouldn't get the `no_snapshots` attribute since they have more than one child.
3. `client_unplug_nat` and `server_unplug_nat` should get the `no_snapshots` attribute, if the `client_prepare` and `server_prepare` tests are to stay cached most of the time. If they tend to lose the cache frequently, we should leave things as they are.
4. The `install_ubuntu` tests are very long to run. We should probably leave the hypervisor snapshots for them even though we're not going to restore their results often. It is better to lose a little disk space but spare ourselves the Ubuntu Server installation re-runs if something went unexpected.
5. The `install_guest_additions` tests may be marked with the `no_snapshots` attribute, no big harm.

After these optimizations we're going to get a pretty good balance between saving the disk space and saving the time for the test runs. A lot of preparatory tests have got the `no_snapshots` attribute, because we assume that they are not going to be run too often (just one time, ideally). The `client_prepare` and `server_prepare` tests are considered the "anchor" tests: we assume that their results will be often used when running the "actual" complex tests, which are going to be run much more often.

The rules above are not universal and you should just keep them in mind as a general approach. Of course there're situations when other rules should be applied, so don't be afraid to experiment!

## Conclusions

In Testo-lang the `no_snapshots` feature allows you to save some disk space, but potentially compromises the tests running time. However, if this feature is well-applied, the damage to run time might be insignificant or just nonexistent at all. So before appliying this feature you should consider which tests are going to be run often and which are going to be cached most of the time.

You can find the complete test scripts [here](https://github.com/testo-lang/testo-tutorials/tree/master/qemu/11%20-%20no_snapshots).
