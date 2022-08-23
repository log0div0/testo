# Params

## Param declaration

Params let you simplify the test scripts management by naming some string constants as global-visible identifiers. You can use those identifiers to access the string constants in the test scripts.

Param (global string constants) declaration is done with the `param` keyword. The declaration has the following syntax:

```text
param <name> <value>
```

- `name` - Type: identifier. The name for the param. Must be uniqie.
- `value` - Type: string. Param value.

After the declaration, the param becomes accessible inside test scripts. You can access other params while declaring a new param.

It is also possible to declare params with the `--param <param_name> <param_value>` command line arguments. These kind of params are also accessible inside the test scripts, like the static-declared params.

> Command-line argument `param` names must not conflict with the static-declared params' names.

> A param is available inside the whole test script. Even if referencing occurs earlier that the declaration.

Once declared, a param could not change its value. An error will be generated if such an attempt is done.

### Example

Let's assume that there is a command-line argument `--param ISO_DIR /path/to/iso`  passed to the Testo interpreter. Then

```testo
param param1 "value1"  # translates to "value1"
param param2 "${ISO_DIR}/${param1}" # translates to "/path/to/iso/value1"
```

## Special (reserved) params

Some params are reserved and play a role of actions default behaviour configuration. Such params let you adjust the default timeouts and intervals in some actions. For example, the default `wait` timeout is 1 minute. If you want to change the default timeout value for the `wait` action (to 2 minutes for example) you need to declare `TESTO_WAIT_DEFAULT_TIMEOUT` param like this:

```testo
param TESTO_WAIT_DEFAULT_TIMEOUT "2m"

test my_test {
    my_vm wait "Hello world" # Default timeout is 2m
}
```

> Special params could be defined only one time.

> Special params fall under the same rules as usual params. For example, it is possible to define a special param with the command line argument (`--param`).

### Special params list

- `TESTO_WAIT_DEFAULT_TIMEOUT` - default timeout in `wait` actions.
- `TESTO_WAIT_DEFAULT_INTERVAL` - default time interval between screen checks in `wait` actions.
- `TESTO_CHECK_DEFAULT_TIMEOUT` - default timeout in `check` conditions.
- `TESTO_CHECK_DEFAULT_INTERVAL` - default time interval between screen checks in `check` conditions.
- `TESTO_MOUSE_MOVE_CLICK_DEFAULT_TIMEOUT` - default timeout for `mouse` actions (if a timeout is applicable).
- `TESTO_PRESS_DEFAULT_INTERVAL` - default time intervals between keys pressings in `press` actions.
- `TESTO_TYPE_DEFAULT_INTERVAL` - default time intervals between keys pressings in `type` actions.
- `TESTO_EXEC_DEFAULT_TIMEOUT` - default timeout in `exec` actions.
- `TESTO_COPY_DEFAULT_TIMEOUT` - default timeout in `copyto` and `copyfrom` actions.
- `TESTO_SHUTDOWN_DEFAULT_TIMEOUT` - default timeout for `shutdown` action.
- `TESTO_DISK_DEFAULT_BUS` - default value of a `bus` attribute in a [disk configuration](Machines.md#disks-configuration).
- `TESTO_SNAPSHOT_DEFAULT_POLICY` - default value of a `snapshots` attribute in a [tests declaration](Tests.md#tests-without-hypervisor-snapshots).

## Param referencing

### Syntax

To reference a param, the following syntax is used:

```testo
"${VAR_REF}"
```

where `VAR_REF` is the param's name.

You could put a param referencing in any part of a string and it will be OK:

```testo
"Hello ${VAR_REF}"
```

Params are avaliable basically in any string in Testo-lang. You could use them inside multiline strings, wait-expression, JS-selections, virtual entities declarations and so on

```testo
"""Hello
	${VAR_REF}
"""

js """
	return find_text("${Menu entry}").match_color("${foreground_colour}, "${gray}")
"""

machine my_ubuntu {
	...
	iso: "${ISO_DIR}/ubuntu_server.iso"
}
```

> You can escape the referencing characters `${` with another dollar sign: `$${`

Aside from params, you can reference counter values inside `for` loops and macro arguments inside macro bodies. The syntax remains the same.

### Resolve order

When you're referencing an identifier, its value is resolved in a specific order. If the value is found at any step the algorithm stops:

1. If the referenced identifier is a `counter` value inside a `for` loop, then the counter's value is returned.
2. If the referenced idenfifier is an argument value inside a `macro` body, then the argument's value is returned.
3. If the referenced identifier is a static-declared or a command-line declared `param`, its value is returned.
4. An error is generated since Testo couldn't find the identifier's value.

You can check that a param is declared with a `DEFINED` expression in an [`if`](Conditions.md) statement:

```testo
if (DEFINED VAR_REF) {
  print "var ref is ${VAR_REF}"
} else {
  print "var ref is not defined"
}
```
