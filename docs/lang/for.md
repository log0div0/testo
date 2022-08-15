# Cycles

Testo-lang supports cycle management with a `for` clause. The `for` loops have the next syntax:

```text
for (<counter> IN <range>) {
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

## Cycle header

The `for` clause header consists of the `<counter>` declaration (which must be an identifier) followed by the key-word `IN` and counter's values definition with the `range` construct. The `range` section describes how many iterations would the cycle have. The `range` could be specified with one the following ways:

1.	`RANGE <start> <finish>`, where the `start` and `finish` are **non-negative integers** or **strings**. If a string is used, the value inside the string must be convertible to a non-negative integer. Inside the string [param referencing](param#param-referencing) is available. The `finish` value must be **greater** than the `start` value. During the cycle run, the `counter` will take all the consecutive integer values, starting from the `start` value (included) and finishing with the `finish - 1` value (included).
2.	`RANGE <finish>`. This may be considered as a special case for the `RANGE <start> <finish>`, with the `start` is set to `0` by default.

RANGE examples:

```testo
RANGE 5 10 # The counter will take values 5, 6, 7, 8, 9
RANGE "5" "10" # The same as above
RANGE 5 # The counter will take values 0, 1, 2, 3, 4
RANGE "${max_iterations}" # The counter will take values based on the the "max_iterations" param value.
```

## Cycle body

Inside the cycle body you can use all the language constructs, which are available inside a command body: actions, conditions, macro calls. Additionally there are special cycle-control statements available: `continue` (go the the next iteration) and `break` (exit the cycle).

It is possible to specify a non-mandatory `else` clause for a cycle. The else clause executes after all the iterations complete normally. This means that the cycle run did not encounter a `break` statement.

Inside the loop you can access the counter's value like any param value.

**Example**

```testo
test some_test {
	some_vm {
		for (i IN RANGE "5" "100") {
			if ("${i}" EQUAL "10") {
				continue
			}
			print "${i}"
			if ("${i}" EQUAL "20") {
				break
			}
		} else {
			print "All the loops worked without a break"
		}
	}
}
```

Keep in mind that in this example control never reaches the `else` clause, because on the 15th iteration the `break` statement is reached.
