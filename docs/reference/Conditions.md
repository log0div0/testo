# Conditions

Testo language supports condition management with an `if` clause. If-statements can be placed at the same level as actions - e.g. in [commands](Tests.md#commands-syntax) body. If-statements have the next syntax:

```text
if (expression) {
	action1
	action2
	action3
	...
} [else {
	action1
	action2
	...
}]
```

The syntax is pretty much straightforward: an `if` clause consists of an `expression` which may be evaluated into either `true` or `false`. If the expression value is `true`, the main action block is run. If the exression is equal to `false`, then the `else` action block is reached (if there is any `else` clause).

Inside both action blocks ("main" and "else") all the actions, nested conditions and loops are allowed.

Conditions may be stacked as follows:

```text
if (expression) {

} else if {

} else if {

}
...
```

## Expressions

Expressions are entities which could be evaluated into a `true` or a `false`. At the moment expressions are used only inside the `if` clause.

Possible expressions in Testo-lang are listed below:

- A string. An empty (zero-length) string is treated as a `false`, othwerwise as a `true`.
- A negation (`NOT <expr>`). Logical complement to the `expr`.
- `DEFINED var`. Returns `true` if the `var` param is defined. Which means its value could be obtained, either from the static declaration, or from the command line argument `--param` (even if the value is an empty string). False otherwise.
- Comparisons. Described below.
- Logical conjunction (`<expr1> AND <expr2>`). Returns `true` if both `expr1` AND `expr2` are `true`. False otherwise.
- Logical disjunction (`<exrp1> OR <expr2>`). Returns `true` if one or both of `expr1` and `expr2` are `true`. False otherwise.
- **Only for if statements placed in virtual machines commands**. Checking the contents of screen with select expressions (`check <select_expr> [timeout timeout_spec] [interval interval_spec]`). For `select_expr` specifications see [here](Actions.md#wait). The checks go on until the `select_expr` is found (`check` result is `true`) or the timeout is triggered (`check` result is `false`). Default timeout is `1ms` which means just one instant check will be performed. You can adjust the checks frequency with the `interval` argument (default value is `1s`).

You can use parentheses to group up the expressions.

### Comparisons

Testo lang allows you to compare **strings**. The comparison result is a boolean (`true` or `false`).

Possible comparison types in Testo lang:

- `<string1> STREQUAL <string2>` - returns `true` if both strings are the same. False otherwise.
- `<string1> STRLESS <string2>` - returns `true` if `string1` lexicographically less than `string2`. False otherwise.
- `<string1> STRGREATER <string2>` - returns `true` if `string1` lexicographically greater than `string2`. False otherwise.
- `<string1> EQUAL <string2>` - applicable if both `string1` and `string2` are convertible to integers. Returns `true` if after conversion to integers both operands will be equal. False otherwise. If `string1` or `string2` are not convertible to integers, an error will be generated.
- `<string1> LESS <string2>` - applicable if both `string1` and `string2` are convertible to integers. Returns `true` if after conversion to integers left operand is less than the right. False otherwise. If `string1` or `string2` are not convertible to integers, an error will be generated.
- `<string1> GREATER <string2>` - applicable if both `string1` and `string2` are convertible to integers. Returns `true` if after conversion to integers left operand is greater than the right one. False otherwise. If `string1` or `string2` are not convertible to integers, an error will be generated.

### Examples

```testo
"SOME STRING" # true
"" # false
"${some_var}" # The value depends on the some_var param. An error will be generated if the some_var param is not defined
DEFINED some_var # true if some_var is defined (even if the value is a zero-length string)
check "Hello world" # true, if there is a text "Hello world" on the screen
"5" EQUAL "${some_var}" # true, if "${some_var}" has the value "5"
"ABC" STRLESS "BCD" # true
NOT ("ABC" STRLESS "BCD") # false
NOT ("ABC" STRLESS "BCD") OR "5" EQUAL "5" # true
NOT ("ABC" STRLESS "BCD" AND "5" EQUAL "5") # false
```
