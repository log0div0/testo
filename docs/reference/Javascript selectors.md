# Main concepts

JS-selectors come in handy when you want to perform an especially tricky mouse move action or to wait a really complex event to appear on the screen. Basically JS-selectors let you extend default Testo-lang possibilities and write your own screen-searching logic, no matter how elaborate.

Javascript selectors can be used in two situations in Testo-lang:

1. To specifiy a complex screen event for a `wait` action and a `check` conditional expression.
2. To specify a complex place to move the mouse cursor for the `mouse` actions.

JS-selectors start with a keyword `js`, followed by a string (one-line or multiline) containing the Javascript snippet. Inside the snippet you can use built-in functions and objects which are described later in this documentation. Those functions will provide you with the information about the current screen's contents.

> If a JS-selector is placed inside a one-line string, all the double quotes in the selector must be escaped with the `\` character. If the selector is placed in a multiline string, there's no need to escape double quotes.

There are some differences between snippets used for the `wait` (and `check`) and for the `mouse` acitons.

## JS-selectors for the `wait` and `check`

When used in the [`wait`](Actions.md#wait) actions and the [`check`](Conditions.md#expressions) conditional expressions, JS-selectors must return a **boolean**: `true` or `false`.

Returning a `true` means that the specified event is found on the screen and the waiting (checking) must be finished. A [`check`](Conditions.md#expressions) expression returns `true` in this situation.

Returning a `false` means that the event hasn't happened on the screen yet and the searching must continue. In this case the interpreter will sleep for the `time_interval` specified for the [`wait`](Actions.md#wait) or [`check`](Conditions.md#expressions) action. After that it tries to process the JS-selector one more time with another screenshot.

If the `timeout` time interval is exceeded, a [`wait`](Actions.md#wait) action generates an error and the test fails. A [`check`](Conditions.md#expressions) expression returns `false` in this situation.

## JS-selectors for the `mouse` actions

When used in the [`mouse`](Mouse%20actions.md) actions, JS-selectors must return a **Point**: an object (of any class, it doesn't matter) with the `x` and `y` properties with numbers as their values.

Returning a point means that the JS-selector found the destination to move the cursor to. So the [`mouse`](Mouse%20actions.md) action finishes waiting and moves the cursor to the returned point.

If the JS-selector is not ready to return a point (the expected event hasn't happend on the screen yet) it should throw [`ContinueError`](#ContinueError), which will force the [`mouse`](Mouse%20actions.md) action to sleep a little bit and try to process the JS-selector one more time with another screenshot.

If a [`mouse`](Mouse%20actions.md) action contains a JS-selector and the processing time of this selector exceeds the [`mouse`](Mouse%20actions.md) action timeout (1 minute by default), then a timeout error will be generated.

# Built-in global functions

## print(arg1, arg2, ...)

Prints all the arguments to the stdout. Could come handy when debugging.

**Arguments**: an arbitrary number of arguments of any types

## find_text(text)

Finds and returns all the textlines with the value `text` (case sensitive) on the virtual machine screen. A textline is a sequence of characters aligned horizontally. Textlines where characters have big horizontal space between them are consdired different textlines. For example, if there are 3 textlines "Install the software", "Install" and "Do not install" on the screen, then `find_text("Install")` returns new `TextTensor` with two 2 instances of the same text: "Install" (one from "Install" and the other from "Install the Software"). "Do not install" won't get into new Tensor because "install" does not match "Install" (strictly speaking it depends on the font of the text, on how difficult to make distinction between uppercase I and lowercase i). Both instances will have the same text but different coordinates.

**Arguments**:

- `text` - the text to match.

**Return value** - an object of the class [`TextTensor`](#TextTensor), with an array containing all the matched text lines.

## find_text()

Same as `fine_text(text)`, but returns all the textlines found on the screen.

**Return value** - an object of the class [`TextTensor`](#TextTensor), with an array containing all the text lines on the screen.

## find_img(path_to_template)

Finds and returns all the images matching the template located in `path_to_template`.

**Arguments**:

- `path_to_template` - path to the template file on the disk.

**Return value** - an object of the class [`ImgTensor`](#ImgTensor), with all the images found matching the specified template.

# Exceptions

## ContinueError

Interrupt the processing of the current JS-selector and try again a bit later. May be applied only inside the `mouse` actions (not applicable in the `wait` and `check` expressions).

# Class Tensor

It's a base class for TextTensor and ImgTensor

## Methods

### from_top(index)
### from_bottom(index)
### from_left(index)
### from_right(index)

Select the textline with the specified index from the array of textlines sorted from top to bottom.

**Arguments**:

- `index <integer>` - the index of the textline to be selected. With the `index == 0` the "uppermost" textline on the screen will be selected.

**Return value** - new object of `TextTensor`, containing just one selected textline.

> Textlines are sorted by the Y coordinate of the top edge of the textlines.

### center()
### left_bottom()
### left_center()
### left_top()
### center_bottom()
### center_top()
### right_bottom()
### right_center()
### right_top()

Return a Point with the coordinates of the geometric center of the textline.

**Arguments**: no

**Return value** - new object `Point`, containing the coordinates of the geometic center of the textline.

> This method is accessible only when the current `TextTensor` contains exactly one textline. Otherwise an error is generated.

## Properties

### `x <integer>`
### `y <integer>`

The X or Y coordinate value (the bigger `x` or `y` means more towards the right/bottom).

> This property is accessible only when the current `TextTensor` contains exactly one textline. Otherwise an error is generated.

# Class TextTensor

Class `TextTensor` contains the information about strings on the screen. Could be obtained with a [`find_text`](global_funcs#find_text()) global function call.

## Methods

### match_text(value)

Return all the textLine instances on the screen matching the `value` string (case sensitive). For example, if an object of `TextTensor` returned by `find_text()` contains 3 textlines "Install the software", "Install" and "Do not install", then after the `match_text("Install")` new `TextTensor` will be returned with two 2 instances of the same text: "Install" (one from "Install" and the other from "Install the Software". "Do not install" won't get into new Tensor because "install" does not match "Install"). Both instances will have the same text but different coordinates.

**Arguments**:

- `value <string>` - the string value to match.

**Return value** - new object of `TextTensor`, with the strings matching the `value` string.

### match_color(foreground_color, background_color)

Return all the textline instances on the screen with the letters color matching the `foreground_color` and the background color matching the `background_color`. Possible colors are: white, gray, black, red, orange, yellow, green, cyan, blue, purple.

**Arguments**:

- `foreground_color <string>` -  The letters color to match.
- `background_color <string>` -  The background color to match.

**Return value** - new object of `TextTensor` with the matching strings.

### match_foreground(value)

Return all the textlines with color of letters matching the `value`. The same as `match_color(value, null)`.

**Arguments**:

- `value <string>` - The color of letters to match.

**Return value** - new object of `TextTensor`, with the strings whose letters color match the `value`.

### match_background(value)

Return all the textline instances on the screen with background color matching the `value`. The same as `match_color(null, value)`.

**Arguments**:

- `value <string>` -  The background color to match.

**Return value** - new object of `TextTensor` with the strings which background color matches the `value`.

# Class ImgTensor

Class `ImgTensor` contains the information about images on the screen. Could be obtained with a [`find_img`](global_funcs#find_img()) global function call.

# Class Point

The `Point` class contains the information about a point on the virtual machine screen.

## Methods

### move_up(N) / move_down(N) / move_right(N) / move_left(N)

Returns a new Point with the `y` coordinate reduced by N pixels compared to the current object.

**Arguments**:

- `N <integer>` - Number of pixels for the new Point to be "higher" than the current Point object.

**Return value** - a `Point` object with the new coordinates.

## Properties

### `x <integer>`

The X coordinate value (more `x` means more to the right).

### `y <integer>`

The Y coordinate value (more `y` means more to the bottom).
