# Testo lang basic syntax

## Literals

There are following literal types available in Testo lang:

-   Integer literals.
-   One-line string literals.
-   Multiline string literals.
-   Boolean literals.
-   Memory size literals.
-   Time interval literals.

### Integer literals

An integer literal is a contiguous sequence of numbers. Optionally the literal can be prefixed with a "+" or a "-" sign.

### String literals

A **One-line string** is an arbitrary sequence of characters encased in double quotes (for example,`"hello world!"`). It is possible to use newlines with an escape character inside one-line strings, like this:

```testo
"This is an example \
of some \
multiline string"
```
A **multiline string** is an arbitrary sequense of charachters encased in triple double qoutes `"""`. It is possible to use newlines without any escape characters inside miltiline-strings, like this:

```testo
"""This is an example
of some
multiline string
"""
```

Inside strings (one-line and multiline alike) you can use the escape character `\`. For example, the string `"This is \"Hello World\" example"` will be translated to `This is "Hello World" example`.

However, it is not necessary to escape double quotes in multiline strings. You only have to escape triple double quotes in multiline strings (`\"`).

```testo
"""This is an example
"Hello world"
of multiline string
"""

"""This is an example
\"""Triple qouted part\"""
of multiline string
"""
```

You can [reference](param) params inside strings. For example,

```testo
"""This is an example of ${VAR_REF}
string concatenation
"""
```

with `VAR_REF`'s value "some value" will be translated to

```testo
"""This is an example of some value
string concatenation
```

### Boolean literals

**Boolean literals** are just two keyword-reserved identifiers: `true` and `false`. Boolean literals are to be used as attributes' values. They are not allowed in `if` statements' expressions.

### Memory size literals

A **memory size literal** has the following syntax: `Positive integer + Memory Specifier`. Possible memory specifiers are `Kb` (Kilobytes), `Mb` (Megabytes) and `Gb` (Gigabytes). Examples: `512Mb`, `3Gb`, `640Kb`.

### Time interval literals

A **time interval literals** has the following syntax: `Positive integer + Time Specifier`. Possible time specifiers are `ms` (milliseconds), `s` (seconds), `m` (minutes), `h` (hours). Examples: `600s`, `1m`, `5h`, `50ms`.

## Identifiers

An identifier must start with a letter or with an `_` symbol. The following symbols must consist of letters, numbers, `-` or `_` symbols. Identifiers are used as names in virtual machines, flash drives, networks, params, tests and macros declarations.

Correct examples: `example`, `another_example`, `_this_is_good_too`,
`And_even-this233-`

Invalid identifiers: `example with spaces`, `5example`

## Key words

Some identifiers are reserved as key words. It is not allowed to use them when naming entities.

- `abort` - Action "abort the test".
- `print` - Action "print a message".
- `type` - Action "type a text on the virtual keyboard".
- `wait` - Action "wait an event to happen on the virtual machine screen".
- `sleep` - Action "sleep for some time".
- `mouse` - Mouse-related action.
- `move` - Action "move the virtual mouse cursor".
- `click` - Action "press the left mouse button".
- `lclick` - The same as `click`.
- `rclick` - Action "press the right mouse button".
- `dclick` - Action "double-click the left mouse button".
- `hold` - Action "press down and hold a button".
- `release` - Action "release a button".
- `lbtn` - Left mouse button specifier for `mouse hold` actions.
- `rbtn` - Right mouse button specifier for `mouse hold` actions.
- `check` - Check for an event on the virtual machine screen.
- `press` - Action "press a keyboard button".
- `plug` - Action "plug a device".
- `unplug` - Action "unplug a device".
- `start` - Action "power on a virtaul machine".
- `stop` - Action "power off a virtual machine".
- `shutdown` - Action "send the SIGTERM signal to the virtual machine OS".
- `exec` - Action "execute a script on the virtual machine".
- `copyto` - Action "copy files from the Host to a virtual machine or a flash drive".
- `copyfrom` - Action "copy files from a virtual machine or a flash drive to the Host".
- `timeout` - Timeout specifier for some actions.
- `interval` - Time between iterations for some actions.
- `test` - Beginning of a test declaration.
- `machine` - Beginning of a virtual machine declaration.
- `flash` - Beginning of a virtual flash drive declaration.
- `network` - Beginning of a virtual network declaration.
- `param` - Beginning of a param declaration.
- `macro` - Beginning of a macro declaration.
- `dvd` - DVD-drive specifier for the actions `plug` and `unplug`.
- `if` - Beginning of an `if` statement.
- `else` - Beginning of an optional `else` clause for `if` and `for` statements.
- `for` - Beginning of a `for` statement.
- `IN` - Is used before the `<range>` in the `for` header.
- `RANGE` - Beginning of a range.
- `break` - "Exit the cycle" statement.
- `continue` - "Go to the next cycle iteration" statement.
- `include` - Include another .testo file directive.
- `js` - The beginning of a javascript-snippet for a `wait` if `check`.
- `img` - Follows `wait` and `check` when you want to detect an image on the screen.
- `DEFINED` - Checking if a param is defined.
- `LESS` - "Less" comparison for two integer-convertible strings.
- `GREATER` - "Greater" comparison for two integer-convertible strings.
- `EQUAL` - "Equality" comparison for two integer-convertible strings.
- `STRLESS` - Lexicographical "less" comparison for two strings.
- `STRGREATER` - Lexicographical "greater" comparison for two strings.
- `STREQUAL` - Lexicographical "equality" comparison for two strings.
- `NOT` - Negation for an expression in `if` statements.
- `AND` - Logical conjunction for expressions in `if` statements.
- `OR` - Logical disjunction for expressions in `if` statements.
- `true` - Logical one.
- `false` - Logical zero.

## Keyboard key literals

- ESC
- ONE
- TWO
- THREE
- FOUR
- FIVE
- SIX
- SEVEN
- EIGHT
- NINE
- ZERO
- A
- B
- C
- D
- E
- F
- G
- H
- I
- J
- K
- L
- M
- N
- O
- P
- Q
- R
- S
- T
- U
- V
- W
- X
- Y
- Z
- MINUS
- EQUALSIGN
- BACKSPACE
- TAB
- LEFTBRACE
- RIGHTBRACE
- ENTER
- LEFTCTRL
- SEMICOLON
- APOSTROPHE
- GRAVE
- LEFTSHIFT
- BACKSLASH
- COMMA
- DOT
- SLASH
- RIGHTSHIFT
- LEFTALT
- SPACE
- CAPSLOCK
- F1
- F2
- F3
- F4
- F5
- F6
- F7
- F8
- F9
- F10
- F11
- F12
- NUMLOCK
- KP_0 (Num Pad 0)
- KP_1 (Num Pad 1)
- KP_2 (Num Pad 2)
- KP_3 (Num Pad 3)
- KP_4 (Num Pad 4)
- KP_5 (Num Pad 5)
- KP_6 (Num Pad 6)
- KP_7 (Num Pad 7)
- KP_8 (Num Pad 8)
- KP_9 (Num Pad 9)
- KP_PLUS (Num Pad +)
- KP_MINUS (Num Pad -)
- KP_SLASH (Num Pad /)
- KP_ASTERISK (Num Pad \*)
- KP_ENTER (Num Pad Enter)
- KP_DOT (Num Pad .)
- SCROLLLOCK
- RIGHTCTRL
- RIGHTALT
- HOME
- UP
- PAGEUP
- LEFT
- RIGHT
- END
- DOWN
- PAGEDOWN
- INSERT
- DELETE
- LEFTMETA
- RIGHTMETA
- SCROLLUP
- SCROLLDOWN
