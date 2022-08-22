# Actions

This pages contains the list of all available actions that can be applyed to virtual machines. Some of them can also be applyed to virtual flash drives, to be more precise they are `copyto`, `copyfrom`, `abort` and `print`.

## start

Starts a virtual machine. The virtual Machine must be stopped at the moment of the call. Starting an already started virtual machine results in an error.

```text
start
```

## stop

Stops a virtual machine, simulating power failure. The virtual machine must be running at the moment of the call. Trying to stop an already stopped virtual machine results in an error.

```text
stop
```

## shutdown

Sends the ACPI signal to a virtual machine, launching a "mild" virtual machine stop. The action waits for the virtual machine to get powered off. Waiting timeout depends on the optional argument `timeout_time_spec`. Virtual machine OS must support ACPI signal processing to make this action work. The virtual machine must be running at the moment of the action call. Trying to shutdown an already stopped virtual machine results in an error.

```text
shutdown [timeout timeout_time_spec]
```

**Arguments**:

- `timeout_time_spec` - Type: time interval or string. Timeout for the virtual machine to stop. Default value - 1 min. If the string type is used, the value inside the string must be convertible to a time interval. Inside the string [param referencing](Params.md#param-referencing) is available.

**Examples**:

```testo
  shutdown

  shutdown timeout 10m

  shutdown timeout "5m"

  #works if the value of param shutdown_timeout is convertible to a time interval
  shutdown timeout "${shutdown_timeout}"
```

## press

Sends signals to a virtual machine, providing pressings of keyboard keys specified in `key_spec`. The `press` action also supports sending sequences of key pressings. In this case the key specifications must be divided by commas. You can adjust the sleep time interval between the pressings with the `interval_time_spec` interval. The virtual machine must be running.

```text
press <key_spec1>[,key_spec2][,key_spec3]... [interval interval_time_spec]
```

**Arguments**:

- `key_spec` - Key specification to press.
- `interval_time_spec` - Type: time interval or string. Sleep time interval between the key pressings. Default value: `30ms`. If the string type is used, the value inside the string must be convertible to a time interval. Inside the string [param referencing](Params.md#param-referencing) is available. Default value can be changed with the `TESTO_PRESS_DEFAULT_INTERVAL` param. See [here](Params.md#special-reserved-params) for more information.

### Key specification

Key specification is a special language clause and looks like this:

```text
<key_id1>[+key_id2][+key_id3]...[*number]
```

`key_id` is an identifier containing the name of a key. See [here](Language%20lexical%20elements.md#keyboard-key-literals) for a complete list of key values. `key_id` is **case-insensitive**, so for example `enter` is treated the same way as `Enter` or `ENTER`.

Key specification consists of two parts: combination of the keyboard keys **pressed simultaneously** and the number of times to press this combination. The combination part must contatin at least one `key_id`. To press several keys simultaneously you need to add more `key_ids` and divide keys with the "+" sign. The number part shows how many times the combination part must be pressed.

The number part (`number`) may be either a positive integer or a string. If the string type is used, the value inside the string must be convertible to a positive integer. Inside the string [param referencing](Params.md#param-referencing) is available.

**Key spec examples:**

- `Down` - press "Down arrow" key one time.
- `LEFTCTRL + alt + Delete` - press keys Ctrl Alt Delete simultaneously one time.
- `LEFTCTRL + alt + Delete * 1` - the same as the previous example.
- `Backspace * "6"` - press the Backspace 6 times.
- `leftalt + F2 * "${number_num}"` - press LeftAlt and F2 keys simultaneously as many times as specified in `number_num` param. Works only if the `number_num` param can be converted to a positive integer.

**`press` action examples**

```testo
  # press "Down" 6 times, then press Enter 1 time
  press Down*6, Enter

  # send key combination Ctrl Alt Delete, then press Down 2 times, then Enter 3 times
  press LeftCTRL + LEFTALT + Delete, Down*2, Enter*3
```

## hold

Hold down keyboard keys specified in `key_spec` in a virtual machine. The keys will be held until explicit call of the [`release`](#release) action. The virtual machine must be running.

```text
hold <key_spec>
```

**Arguments**:

- `key_spec` - Key specification to hold down.

**Notes**

The `hold` action has some restrictions:

1. You can't hold down keys already being held;
2. You can't finish a test with any keys being held. You must call the [`release`](#release) action before the end of the test;
3. You can't use keys being held down in [`press`](#press) actions;
4. You can't use keys being held down when typing text in [`type`](#type) actions;
5. You can't use a number of times in the `key_spec`. It is implied that keys can be held down only one time.

## release

Release keyboard keys specified in `key_spec` in a virtual machine. The keys to be released must be held down by a previously called [`hold`](#hold) action.  If no `key_spec` is specified, all the held keys are to be released.  The virtual machine must be running.

```text
release [key_spec]
```

**Arguments**:

- `key_spec` - Key specification to release.

**Notes**

The `release` action can be used with no arguments. In this case all the keys being held down are to be released. If you need to release only some of the held down keys you have to specify the `key_spec` argument.

The `release` action has some restrictions:

1. You can't release the keys that are not being held down;
2. You can't call the `release` action without a `key_spec` if there is no keys being held down;
3. You can't use a number of times in the `key_spec`. It is implied that keys can be released only one time.

**Examples**

```testo
# Hold down left ctrl and left alt
hold LeftCtrl + LeftAlt

press Delete

# Release only left ctrl
release LeftCtrl

# Release all the keys being held down (e.g. left alt)
release
```

## type

Type a text specified in `text` using the virtual machine keyboard. The virtual machine must be running. All the newline characters ('\n') in the specified text are transformed to the `Enter` key pressings. All the tab characters ('\t') a transformed to the `Tab` key pressings.

```text
type <text> [interval interval_time_spec] [autoswitch <autoswitch_key_spec>]
```

**Arguments**:

- `text` - Type: string. The text to type. Inside the string [param referencing](Params.md#param-referencing) is available.
- `interval_time_spec` - Type: time interval or string. Sleep time interval between the key pressings when typing the text. Default value: `30ms`. If the string type is used, the value inside the string must be convertible to a time interval. Inside the string [param referencing](Params.md#param-referencing) is available. Default value can be changed with the `TESTO_TYPE_DEFAULT_INTERVAL` param. See [here](Params.md#special-reserved-params) for more information.
- `autoswitch_key_spec` - Type: key combination. Enables keyboard layout autoswitching mode (see below). The key combination that Testo will press when trying to change the current keyboard layout.

**Keyboard layout autoswitching mode**

There are situations when you need to type a text containing both English and Russian letters. Prior to Testo Framework 3.0.0, the only option to do it was to emulate a real person's actions, which is not very convenient:

```testo
type "Hello "
press LeftShift + LeftAlt
type "Мир!"
```

Starting with Testo 3.0.0 you can type this kind of text much easier:

```testo
type "Hello Мир!" autoswitch LeftShift + LeftAlt
```

The algorithm here is the following:

1. Before typing the multi-layout text, Testo tries to estimate the current keyboard layout. To do that, Testo would type a few symbols and constantly check the screen state, waiting for enough information to make the decision about the current layout. If Testo failed to estimate the current layout in 10 attemts, an error would be generated.
2. Testo starts typing the text, switching automatically between layouts when necessary using the key combination specified in `autoswitch_key_spec`.

> Trying to type multi-layout text without `autoswitch` keyword will lead to an error.

**Examples**:

```testo
type "Hello world"

type "Hello ${World}"

type """Hello ${World}
    Some multiline
    string ${World} another multiline
string
"""

type "Hello world" interval 30ms

type "Hello world" interval "1s"

# works if the param value "type_interval" is convertible to a time interval
type "Hello world" interval "${type_interval}"

type "Привет world!" autoswitch LeftShift + LeftAlt
```

## mouse

Mouse-related actions are documented [here](Mouse%20actions.md).

## sleep

Unconditional sleep for specified amount of time.

```text
sleep <timeout timeout_time_spec>
```

**Arguments**

- `timeout_time_spec` - Type: time interval or string. Time period to sleep. If the string type is used, the value inside the string must be convertible to a time interval. Inside the string [param referencing](Params.md#param-referencing) is available.

**Examples**:

```testo
  sleep 10s

  # works if the param value "sleep_timeout" is convertible to a time interval.
  sleep "${sleep_timeout}"
```

## wait

Wait for an event to appear on the virtual machine screen. The event is specified in the `select_expr` argument. Waiting timeout is specified in the `timeout_time_spec` argument. If the expected event does not appear before the timeout, an error is generated. You can adjust the screen check frequency with the `interval_time_spec` argument. The virtual machine must be running.

```text
wait <select_expr> [timeout timeout_time_spec] [interval interval_time_spec]
```

**Arguments**:

- `select_expr` - Select expression (an event) to wait.
- `timeout_time_spec` - Type: time interval or string. Timeout for the `select_expr` to appear on the screen. Default value: `1m`. If the string type is used, the value inside the string must be convertible to a time interval. Inside the string [param referencing](Params.md#param-referencing) is available. Default value can be changed with the `TESTO_WAIT_DEFAULT_TIMEOUT` param. See [here](Params.md#special-reserved-params) for more information.
- `interval_time_spec` - Type: time interval or string. Time interval between the screen checks for the expected event to appear. Default value: `1s`. If the string type is used, the value inside the string must be convertible to a time interval. Inside the string [param referencing](Params.md#param-referencing) is available. Default value can be changed with the `TESTO_WAIT_DEFAULT_INTERVAL` param. See [here](Params.md#special-reserved-params) for more information.

### Select expressions for the `wait` and `check` actions

The `wait` action and `check` expression provide you with not only some basic screen checks (the presense or absense of a text or an image), but also some more coplex checks, as well as whole combinations of checks. There are three types of checks in Testo-lang:

**Basic text checks**

If you just want (or check) to simply wait for a text to appear on the screen, all you need to do is to perform a `wait` action with the following syntax:

```testo
wait "Extected string"
```

Inside the string [param referencing](Params.md#param-referencing) is available, like this:

```testo
wait "Extected string with a param value ${param}"
```

> The text must be continuous, i.e. there must be no newlines or big gaps between characters. For example, if a text consists of three words, but the third word stays too far from the first two words, then the wait for the whole text will fail.

**Basic image checks**

If you need wait (or check) for an image to appear on the screen, then you should place the `img` specifier after the `wait` (or `check`), followed by a path to the image template you want to find:

```testo
wait img "/path/to/img/to/be/searched"
```

**Complex javascript-based checks**

For more elaborate screen checks in the `wait` (or `check`) you can use javascript selections. These selections look like javascript code snippets which must return a bool-value `true` or `false`.

If the javascript returned `true` - then `wait` and `check` actions are considered complete, and test control goes to the next action. Otherwise `wait` and `check` processing continues until a `true` is returned or a timeout is due. Any other returned value is treated as an error and the test will fail.

```testo
wait js "return find_text('Hello world').match_color('blue', 'gray').size() == 1"
```

The example above waits for a text "Hello world" with blue charecters and gray background to appear on the screen. Such a string could represent, for example, a selected menu entry.

For more information about javascript-selections see [here](Javascript%20selectors.md).

**Using several checks together in the same action**

Checks may be combined together into whole select expressions with logical connectives and negations: `&&` (AND), `||` (OR) and `!`(NOT). Exressions also support nested expressions enclosed in parentheses.

**Examples**

```testo
wait "Hello world" && img "${IMG_DIR}/my_icon.png"
```

Wait for simultaious absense of the text "Hello world" and presence of the image, which path depends on the value of the `IMG_DIR` param.

```testo
# works if the param value "wait_timeout" is convertible to a time interval
wait !"Hello world" || js """
   return find_text("Menu entry")
      .match_color("${foreground_colour}", "gray")
      .size() == 1
""" timeout 10m interval "${wait_interval}"
```

Wait for either of two events:
1. Absense of the text `Hello world` on the screen.
2. Presence of the text `Menu entry` on the screen. `Menu entry` must have the grey background and the character color determined by the `foreground_colour` param value.

Check frequency is determined by the value of the `wait_interval` param. Maximum waiting time is 10 minutes.

## plug

The `plug` action allows you to attach various devices to a virtual machine. All the possible ways to use the `plug` actions are listed below.

### plug flash

> This action is not available for Hyper-V

Insert a virtual flash drive into a virtual machine.

```text
plug flash <flash_name>
```

**Arguments**:

- `flash_name` - Type: identifier or string. The name of the virtual flash drive to insert. The flash drive must be declared and not be inserted in any virtual machine. If the string type is used, the value inside the string must be convertible to an identifier. Inside the string [param referencing](Params.md#param-referencing) is available.

**Examples**:

```testo
  plug flash my_flash

  plug flash "my_flash"

  # works if param value "flash_name" is convertible to an identifier
  # and if the flash drive with such name is declared
  plug flash "${flash_name}"
```

> At the moment it is not allowed to have more than 1 flash drive plugged into a virtual machine at the same time.

> All plugged flash drives must be unplugged before the end of the test. It is not allowed to finish a test with plugged flash drives.

> Trying to plug an already plugged flash drive will result in an error.

### plug nic

Insert a Network Interface Card (NIC) into a virtual machine. The name of the NIC must correspond to a name specified in the `nic` attribute of the virtual machine declaration. The virtual machine must be powered off.

```text
plug nic <nic_name>
```

**Arguments**:

- `nic_name` - Type: identifier or string. The name of the NIC to insert. The NIC with this name must be declared in the virtual machine configuration. If the string type is used, the value inside the string must be convertible to an identifier. Inside the string [param referencing](Params.md#param-referencing) is available.

```testo
  plug nic internet_nic

  plug nic "internet_nic"

  # works if param value "nic_name" is convertible to an identifier
  # and if the NIC with the resulting name exists in the virtual machine configuration
  plug nic "${nic_name}"
```

> Plugging/Unplugging NICs requires that the virtual machine must be powered off.

> Trying to plug an already plugged NIC will result in an error.

### plug link

Plug a virtual link to the NIC. The name of the NIC must correspond to a name in the `nic` attribute of the virtual machine declaration.

```text
plug link <nic_name>
```

**Arguments**:

- `nic_name` - Type: identifier or string. The name of the NIC to plug the link into. The NIC with this name must be declared in the virtual machine configuration. If the string type is used, the value inside the string must be convertible to an identifier. Inside the string [param referencing](Params.md#param-referencing) is available.

> Trying to plug an already plugged link will result in an error.

### plug dvd

Insert the iso-image of a disk into the virtual DVD drive of a virtual machine.

```text
plug dvd <path_to_iso_file>
```

**Arguments**:

- `path_to_iso_file` - Type: string. The path to the iso-image. The specified iso-image must exist.

> Trying to plug an iso into the already occupied DVD-drive will result in an error.

### plug hostdev usb

> This action is not available for Hyper-V

Attach a Host-plugged USB device to the virtual machine.

```text
plug hostdev usb <usb_device_address>
```

**Arguments**:

- `usb_device_address` - Type: string. The USB address of the device to be plugged. The address must be represented as `"Bus_num-Device_num"`, where `Bus_num` and `Device_num` are decimal numbers (for instance, `"3-1"`). Inside the string [param referencing](Params.md#param-referencing) is available.

> An attempt to plug an already plugged USB device will result in an error.

> An USB device can't be plugged at more than one Virtual Machine at a time.

> If the test isn't labeled with the `no_snapshots: true` attribute, then all the plugged USB devices must be unplugged before the end of the test. An attempt to finish the test with a USB device attached to a virtual machine will lead to an error.

> You can check the USB address for the device with the `lsusb` utulity (for instance).

> MOST IMPORTANT! Testo does not take snapshots and does not provide any guarantee about the integrity of plugged Host USB devices. Testo won't undo all the changes done to the USB device during tests! Use this action at your own risk!

## unplug

The `unplug` action allows you to detach various devices from a virtual machine. All the possible ways to use the `unplug` actions are listed below.

### unplug flash

> This action is not available for Hyper-V

Remove a virtual flash drive from a virtual machine.

```text
unplug flash <flash_name>
```

**Arguments**:

- `flash_name` - Type: identifier or string. The name of the virtual flash drive to remove. The flash drive must be declared and inserted into current virtual machine. If the string type is used, the value inside the string must be convertible to an identifier. Inside the string [param referencing](Params.md#param-referencing) is available.

> All plugged flash drives must be unplugged before the end of the test. It is not allowed to finish a test with plugged flash drives.

> Trying to unplug an already unplugged flash drive will result in an error.

### unplug nic

Remove a Network Interface Card (NIC) from a virtual machine. The name of the NIC must correspond to a name in the `nic` attribute of the virtual machine declaration. The virtual machine must be powered off.

```text
unplug nic <nic_name>
```

**Arguments**:

- `nic_name` - Type: identifier or string. The name of the NIC to detach. The NIC with this name must be declared in the virtual machine configuration. If the string type is used, the value inside the string must be convertible to an identifier. Inside the string [param referencing](Params.md#param-referencing) is available.

> Plugging/Unplugging NICs requires that a virtual machine must be turned off.

> Trying to unplug an already unplugged NIC will result in an error.

### unplug link

Unplug a virtual link from a NIC. The name of the NIC must correspond to a name in the `nic` attribute of the virtual machine declaration.

```text
unplug link <nic_name>
```

**Arguments**:

- `nic_name` - Type: identifier or string. The name of the NIC to unplug the link from. The NIC with this name must be declared in the virtual machine configuration. If the string type is used, the value inside the string must be convertible to an identifier. Inside the string [param referencing](Params.md#param-referencing) is available.

> Trying to unplug an already unplugged link will result in an error.

### unplug dvd

Remove the current iso-image from the DVD-drive of the Virtual Machine.

```text
unplug dvd
```

> Trying to remove the iso-image from an empty DVD-drive will result in an error.

### unplug hostdev usb

> This action is not available for Hyper-V

Detach a Host-plugged USB device from the virtual machine.

```text
unplug hostdev usb <usb_device_address>
```

**Arguments**:

- `usb_device_address` - Type: string. The USB address of the device to be unplugged. The address must be represented as `"Bus_num-Device_num"`, where `Bus_num` and `Device_num` are decimal numbers (for instance, `"3-1"`). Inside the string [param referencing](Params.md#param-referencing) is available.

> An attempt to unplug a not-plugged USB device will result in an error.

## exec

Execute the specified in the `script` script inside a virtual machine with the interpreter specified in `interpreter`. The `testo-guest-additions` agent must be installed on the virtual machine before calling this action. If the interpreter failed (exit code is not 0), then the current test fails with an error. Stdout and stderr from the `interpreter` are redirected to Testo stdout, therefore you can see the script processing in real time.

```text
exec <interpreter> <script> [timeout timeout_time_spec]
```

**Arguments**:

- `interpreter` - Type: identifier. The name of the interpreter to execute the script. At the moment the next values are allowed: `bash`, `cmd`, `python`, `python2` and `python3`. The interpreter must be installed and available inside the virtual machine OS.
- `script` - Type: string. The script to execute.
- `timeout_time_spec` - Type: time interval or string. Timeout for the script to execute. Default value: `10m`. If the string type is used, the value inside the string must be convertible to a time interval. Inside the string [param referencing](Params.md#param-referencing) is available. Default value can be changed with the `TESTO_EXEC_DEFAULT_TIMEOUT` param. See [here](Params.md#special-reserved-params) for more information.

**Examples**:

```testo
  exec bash "echo Hello world!"

  exec cmd "echo Hello world!" timeout 5m

  # works if the param value "python_timeout" is convertible to a time interval
  exec python """
    print('Hello, world!')
  """ timeout "${python_timeout}"
```

## copyto

Copies a file or a directory from the Host to a virtual machine. The `testo-guest-additions` agent must be installed on the virtual machine before calling this action.

```text
copyto <from> <to> [nocheck] [timeout timeout_time_spec]
```

**Arguments**:

- `from`- Type: string. Path to the file or directory on the Host. The path must exist.
- `to` - Type: string. **Full** destination path on the virtual machine.
- `nocheck` - Type: identifier with fixed value. The presence of this specifier disables the semantic checking of the file existence on the host. This way you can run tests with `copyto` actions even if the `from` file doesn't exist on the host at the moment of running. It is assumed that the file will be there at the actual moment of `copyto` execution.
- `timeout_time_spec` - Type: time interval or string. Timeout for copying to complete. Default value: `10m`. If the string type is used, the value inside the string must be convertible to a time interval. Inside the string [param referencing](Params.md#param-referencing) is available. Default value can be changed with the `TESTO_COPY_DEFAULT_TIMEOUT` param. See [here](Params.md#special-reserved-params) for more information.

> You must specify the full destination path in the `to` argument. For example, if you need to copy the file `/home/user/some_file.txt` on the virtual machine destination `/path/on/vm/some_file.txt` you should call `copyto` like this: `copyto /home/user/some_file.txt /path/on/vm/some_file.txt`. Copying directories falls under the same rules.

> Copying links is not allowed.

> You should keep in mind that with the `nocheck` specifier the `from` files' integrity is not included in the test cache. Which means that changing the `from` files won't lead to the test cache invalidation.

## copyfrom

Copies a file or a directory from a virtual machine to the Host. The `testo-guest-additions` agent must be installed on the virtual machine before calling this action.

```text
copyfrom <from> <to> [timeout timeout_time_spec]
```

**Arguments**:

- `from`- Type: string. Path to the file or directory on the virtual machine. The path must exist.
- `to` - Type: string. **Full** destination path on the Host.
- `timeout_time_spec` - Type: time interval or string. Timeout for copying to complete. Default value: `10m`. If the string type is used, the value inside the string must be convertible to a time interval. Inside the string [param referencing](Params.md#param-referencing) is available. Default value can be changed with the `TESTO_COPY_DEFAULT_TIMEOUT` param. See [here](Params.md#special-reserved-params) for more information.

> You must specify the full destination path in the `to` argument (see `copyto` action notes).

> Copying links is not allowed.

## screenshot

Save the current screen state in a .png file.

```text
screenshot <destination_path>
```

**Agruments**:

- `destination_path` - Type: string. Destination path for the file storing the current screenshot.

> Virtual machine must be powered on at the time of calling this action.

> If the file already exists, it will be overwritten.

## abort

Abort the current test running with an error message. The test is considered failed.

```text
abort <error_message>
```

**Arguments**:

- `error_message` - Type: string. Abort (error) message.

## print

Prints a message to the Testo stdout.

```text
print <message>
```

**Arguments**:

- `message` - Type: string. Message to print.

## repl

Switches the interpreter to the interactive mode. This action is highly useful for writing new tests. You can start with a blank test that consists of the only repl action:

```
test my_new_test {
   repl
}
```

In interactive mode you can type actions (like `type`, `wait` and so on) one-by-one and see the result in real time. Press Ctrl-C to exit interactive mode. The interpreter will print for you the list of succeeded actions which you can copy-paste in the test scenario file.

Apart from that this mode can be useful in debugging purposes as `repl` action can be placed anywhere in the test.

## bug

Add a mention about the bug to the report. This action does not fail the test. It's used mainly in conjunction with JIRA or TFS.

```
bug <bug_id>
```

**Arguments**:

- bug_id  - Type: string. Bug to report about.
