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

### from_top(index)

Select the textline with the specified index from the array of textlines sorted from top to bottom.

**Arguments**:

- `index <integer>` - the index of the textline to be selected. With the `index == 0` the "uppermost" textline on the screen will be selected.

**Return value** - new object of `TextTensor`, containing just one selected textline.

> Textlines are sorted by the Y coordinate of the top edge of the textlines.

### from_bottom(index)

Select the textline with the specified index from the array of textlines sorted from bottom to top.

**Arguments**:

- `index <integer>` - the index of the textline to be selected. With the `index == 0` the "lowermost" textline on the screen will be selected.

**Return value** - new object of `TextTensor`, containing just one selected textline.

> Textlines are sorted by the Y coordinate of the bottom edge of the textlines.

### from_left(index)

Select the textline with the specified index from the array of textlines sorted from left to right.

**Arguments**:

- `index <integer>` - the index of the textline to be selected. With the `index == 0` the most left textline on the screen will be selected.

**Return value** - new object of `TextTensor`, containing just one selected textline.

> Textlines are sorted by the X coordinate of the left edge of the textlines.

### from_right(index)

Select the textline with the specified index from the array of textlines sorted from right to left.

**Arguments**:

- `index <integer>` - the index of the textline to be selected. With the `index == 0` the most right textline on the screen will be selected.

**Return value** - new object of `TextTensor`, containing just one selected textline.

> Textlines are sorted by the X coordinate of the right edge of the textlines.

### center()

Return a Point with the coordinates of the geometric center of the textline.

**Arguments**: no

**Return value** - new object `Point`, containing the coordinates of the geometic center of the textline.

> This method is accessible only when the current `TextTensor` contains exactly one textline. Otherwise an error is generated.

### left_bottom()

Return a Point with the coordinates of the bottom left corner of the textline.

**Arguments**: no

**Return value** - new object `Point`, containing the coordinates of the bottom left corner of the textline.

> This method is accessible only when the current `TextTensor` contains exactly one textline. Otherwise an error is generated.

### left_center()

Return a Point with the coordinates of the center of the left edge of the textline.

**Arguments**: no

**Return value** - new object `Point`, containing the coordinates of the center of the left edge of the textline.

> This method is accessible only when the current `TextTensor` contains exactly one textline. Otherwise an error is generated.

### left_top()

Return a Point with the coordinates of the top left corner of the textline.

**Arguments**: no

**Return value** - new object `Point`, containing the coordinates of the top left corner of the textline.

> This method is accessible only when the current `TextTensor` contains exactly one textline. Otherwise an error is generated.

### center_bottom()

Return a Point with the coordinates of the center of the left edge of the textline.

**Arguments**: no

**Return value** - new object `Point`, containing the coordinates of the center of the left edge of the textline.

> This method is accessible only when the current `TextTensor` contains exactly one textline. Otherwise an error is generated.

### center_top()

Return a Point with the coordinates of the center of the top edge of the textline.

**Arguments**: no

**Return value** - new object `Point`, containing the coordinates of the center of the top edge of the textline.

> This method is accessible only when the current `TextTensor` contains exactly one textline. Otherwise an error is generated.

### right_bottom()

Return a Point with the coordinates of the bottom right corner of the textline.

**Arguments**: no

**Return value** - new object `Point`, containing the coordinates of the bottom right corner of the textline.

> This method is accessible only when the current `TextTensor` contains exactly one textline. Otherwise an error is generated.

### right_center()

Return a Point with the coordinates of the center of the right edge of the textline.

**Arguments**: no

**Return value** - new object `Point`, containing the coordinates of the center of the right edge of the textline.

> This method is accessible only when the current `TextTensor` contains exactly one textline. Otherwise an error is generated.

### right_top()

Return a Point with the coordinates of the top right corner of the textline.

**Arguments**: no

**Return value** - new object `Point`, containing the coordinates of the top right corner of the textline.

> This method is accessible only when the current `TextTensor` contains exactly one textline. Otherwise an error is generated.

## Properties

### `x <integer>`

The X coordinate value (the bigger `x` means more towards the right).

> This property is accessible only when the current `TextTensor` contains exactly one textline. Otherwise an error is generated.

### `y <integer>`

The Y coordinate value (the bigger `y` means more towards the bottom)

> This property is accessible only when the current `TextTensor` contains exactly one textline. Otherwise an error is generated.
