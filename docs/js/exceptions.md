# Exceptions

## ContinueError

Interrupt the processing of the current JS-selector and try again a bit later. May be applied only inside the `mouse` actions (not applicable in the `wait` and `check` expressions).

If a `mouse` action contains a JS-selector with a `ContinueError` exception and the processing time of this selector exceeds the `mouse` action timeout (1 minute by default), then the timeout error will be generated.

**Arguments**: no
