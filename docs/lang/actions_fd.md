# Virtual Flash Drives Actions

> Action with virtual flash drives are not available for Hyper-V

## copyto

Copies a file or a directory from the Host to a virtual flash drive.

```text
copyto <from> <to> [timeout timeout_time_spec]
```

**Arguments**:

- `from`- Type: string. Path to the file or directory on the Host. The path must exist.
- `to` - Type: string. **Full**  destination path on the virtual flash drive.
- `nocheck` - Type: identifier with fixed value. The presence of this specifier disables the semantic checking of the file existence on the host. This way you can run tests with `copyto` actions even if the `from` file doesn't exist on the host at the moment of running. It is assumed that the file will be there at the actual moment of `copyto` execution.
- `timeout_time_spec` - Type: time interval or string. Timeout for the copying to complete. Default value: `10m`. If the string type is used, the value inside the string must be convertible to a time interval. Inside the string [param referencing](param#param-referencing) is available. Default value may be changed with the `TESTO_COPY_DEFAULT_TIMEOUT` param. See [here](param#special-(reserved)-params) for more information.

> You must specify the full destination path in the `to` argument. For example, if you need to copy the file `/home/user/some_file.txt` into the virtual machine with the final destination `/path/on/vm/some_file.txt`, you should call `copyto` like this: `copyto /home/user/some_file.txt /path/on/vm/some_file.txt`. Copying directories falls under the same rules.

> Copying links is not allowed.

> The flash drive must not be plugged in any virtual machine when calling this action.

> You should remember that with the `nocheck` specifier the `from` files' integrity is not included in the test cache. Which means that changing the `from` files won't lead to the test cache invalidation.

## copyfrom

Copies a file or a directory from a virtual flash drive to the Host.

```text
copyfrom <from> <to> [timeout timeout_time_spec]
```

**Arguments**:

- `from`- Type: string. Path to the file or directory on the virtual flash drive. The path must exist.
- `to` - Type: string. **Full** destination path on the Host.
- `timeout_time_spec` - Type: time interval or string. Timeout for copying to complete. Default value: `10m`. If the string type is used, the value inside the string must be convertible to a time interval. Inside the string [param referencing](param#param-referencing) is available. Default value may be changed with the `TESTO_COPY_DEFAULT_TIMEOUT` param. See [here](param#special-(reserved)-params) for more information.

> You must specify the full destination path in the `to` argument (see `copyto` action notes).

> Copying links is not allowed.

> The flash drive must not be plugged in any virtual machine when calling this action.

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

## Macro call

Call the `macro_name` macro. The macro must be declared before the calling. The macro must consist of actions applicable to flash drives.

Macro calls are described [here](macro#macro-call)