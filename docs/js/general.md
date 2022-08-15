# Main concepts

JS-selectors come in handy when you want to perform an especially tricky mouse move action or to wait a really complex event to appear on the screen. Basically JS-selectors let you extend default Testo-lang possibilities and write your own screen-searching logic, no matter how elaborate.

Javascript selectors can be used in two situations in Testo-lang:

1. To specifiy a complex screen event for a `wait` action and a `check` conditional expression.
2. To specify a complex place to move the mouse cursor for the `mouse` actions.

JS-selectors start with a keyword `js`, followed by a string (one-line or multiline) containing the Javascript snippet. Inside the snippet you can use built-in functions and objects which are described later in this documentation. Those functions will provide you with the information about the current screen's contents.

> If a JS-selector is placed inside a one-line string, all the double quotes in the selector must be escaped with the `\` character. If the selector is placed in a multiline string, there's no need to escape double quotes.

There are some differences between snippets used for the `wait` (and `check`) and for the `mouse` acitons.

## JS-selectors for the `wait` and `check`

When used in the [`wait`](/en/docs/lang/actions_vm#wait) actions and the [`check`](/en/docs/lang/if#expressions) conditional expressions, JS-selectors must return a **boolean**: `true` or `false`.

Returning a `true` means that the specified event is found on the screen and the waiting (checking) must be finished. A wait with JS-selector returned a `true` finishes successfully and a `check` expression returns `true`.

Returning a `false` means that the event hasn't happened on the screen yet and the searching must continue. A wait or a check with a JS-selector, which returned `false`,  sleeps for the `time_interval` specified for them. After that it tries to process the JS-selector one more time with another screenshot.

If the `timeout` time interval is exceeded, a wait action generates an error and the test fails. A check expression returns `false` in the same situation.

## JS-selectors for the `mouse` actions

When used in the [`mouse`](/en/docs/lang/mouse) actions, JS-selectors must return a **Point**: an object (of any class, it doesn't matter) with the `x` and `y` properties with numbers as their values.

Returning a point means that the JS-selector found the destination to move the cursor to. So the `mouse` action finishes waiting and moves the cursor to the returned point.

If the JS-selector is not ready to return a point (the expected event hasn't happend on the screen yet) it should throw [`ContinueError`](exceptions), which will force the `mouse` action to sleep a little bit and try to process the JS-selector one more time with another screenshot.

If a `mouse` action contains a JS-selector with `ContinueError` exceptions and the processing time of this selector exceeds the `mouse` action timeout (1 minute by default), then a timeout error will be generated.
