# Built-in global functions

## print(arg1, arg2, ...)

Prints all the arguments to the stdout. Could come handy when debugging.

**Arguments**: an arbitrary number of arguments of any types

**Return value** - no

## find_text(value)

Finds and returns all the textlines with the value `value` (case sensitive) on the virtual machine screen. A textline is a sequence of characters aligned horizontally. Textlines where characters have big horizontal space between them are consdired different textlines. For example, if there are 3 textlines "Install the software", "Install" and "Do not install" on the screen, then `find_text("Install")` returns new `TextTensor` with two 2 instances of the same text: "Install" (one from "Install" and the other from "Install the Software"). "Do not install" won't get into new Tensor because "install" does not match "Install"). Both instances will have the same text but different coordinates.

**Arguments**:

- `value <string>` - the string value to match.


**Return value** - an object of the class [`TextTensor`](text_tensor), with an array containing all the strings with matched value.


## find_text()

Same as `fine_text(value)`, but returns all the textlines found on the screen.

**Arguments**: no


**Return value** - an object of the class [`TextTensor`](text_tensor), with an array containing all the strings on the screen.

## find_img(path_to_template)

Finds and returns all the images matching the template located in `path_to_template`.

**Arguments**:

- `path_to_template <string>` - path to the template file on the disk.


**Return value** - an object of the class [`ImgTensor`](img_tensor), with all the images found matching the specified template.
