# Mouse actions

Mouse actions start with the `mouse` keyword. The exact syntax depends on the mouse action type you want to perform. Most of the `mouse` actions (except for `mouse hold` and `mouse release`) are related to the mouse cursor movement at a specific place on the virtual machine screen. Let's see all the possible ways to specify a place on the screen in Testo-lang.

## How to specify a place on the screen (`destination`)

### Coordinates

The most basic way to specify a point on the scren is with **coordinates**. Coordinates are two numbers separated by a space (`X Y`). The first number is the number of pixels for the X (horizontal) axis, the second number is the number of pixels for the Y (vertical) axis. Point `0 0` is placed at the top left corner of the sreen.

**Example**

```testo
100 250 # A point with the coordinates "100 pixels to the right from the left side of the screen
        # and 250 pixels to the bottom from the top side of the screen
```

### A Text on the screen

It is also possible to move the cursor to the center of a text on the screen (if the text may be found on the screen, naturally). It is possible to use param [referencing](param#param-referencing) inside the text specification. The cursor will be centered on both X and Y axises.

**Examples**

```testo
"My computer" # the center of a "My Computer" string on the screen
"${pic_name}.png"
```

### A text on the screen with additional specifiers

Sometimes a simple centering inside a text on the screen is not enough, and you want to be more specific. For example, there could be several instances of the expected text, and so it is not possible to tell without additional specifications which instance you want to move the cursor to. Or, perhaps, you want to move the mouse cursor not to the center of a text - but to its right edge. Finally, you may want to move the cursor some number of pixels off the text. Testo Framework allows you to add additional specifiers to the expected text and thus adjust the target place precisely.

Specifying the precise destination screen point is a 3-step process:

<img src="/static/docs/lang/mouse_specifiers_1.svg"/>

**Step 1. Selecting the text instance**

When the virtual screen has two or more instances of the expected text, it is mandatory to specify which one you want to move the cursor to. For this purpose, there are four specifiers to your service:

1.  `from_bottom(N)` - select the N-th text instance **from the bottom**. `from_bottom(0)` means the lowermost text instance on the screen.
2.  `from_top(N)` - select the N-th text instance **from the top**. `from_top(0)` means the uppermost text instance on the screen.
3.  `from_left(N)` - select the N-th text instance **from the left**. `from_left(0)` means the leftmost text instance on the screen.
4.  `from_right(N)` - select the N-th text instance **from the right**. `from_right(0)` means the rightmost text instance on the screen.

> If there is only one instance of an expected text on the screen, the first step may be omitted. You may go straight to step 2 or step 3.

> If you try to specify the `from` specifier with the `N` being more than `Expected_text_instances_number_on_the_sreen - 1`, an error will be generated (it's like the array index out of bounds error).

> You can't perform step 1 serveral times. Doing so will result in an error.

**Step 2. Positioning the cursor inside the selected text instance**

By default, the cursor is moved to the center (by X and Y axises) of the selected text instance. However, sometimes it is required to move the cursor to the other part of the text instance. For that purpose you may use the following specifiers:

1.  `left_bottom()` - move the cursor to the bottom left corner of the text.
2.  `left_center()` - move the cursor to the center of the left edge of the text.
3.  `left_top()` - move the cursor to the top left corner of the text.
4.  `center_bottom()` - move the cursor to the center of the lower edge of the text.
5.  `center()` - move the cursor to the center to the X and Y axises center of the text (default behaviour).
6.  `center_top()` - move the cursor to the center of the upper edge of the text.
7.  `right_bottom()` - move the cursor to the bottom right corner of the text.
8.  `right_center()` - move the cursor to the center of the right edge of the text.
9.  `right_top()` - move the cursor to the top right corner of the text.

> You may omit this step if you are okay with the center positioning, which is done by default.

> Trying to perform this step with no `from_` (step 1) specifier will work only if there is only one instance of the expected text.

> You can't perform step 2 serveral times. Doing so will result in an error.

**Step 3. Final cursor positioning**

After the inside-text positioning it is possible to move the cursor to the right, left, up or down for any number of pixels an arbitrary number of times. There are another four specifiers to do so:

1.  `move_left(N)` - move the cursor N pixels to the left.
2.  `move_right(N)` - move the cursor N pixels to the right.
3.  `move_up(N)` - move the cursor N pixels to the top.
4.  `move_down(N)` - move the cursor N pixels to the bottom.

> You can perform step 3 any number of times.

> Trying to perform this step with no `from_` (step 1) specifier will work only if there is only one instance of the expected text.

> Trying to perform this step with no step 2 specifier will mean moving the cursor starting from the center of the text instance.

**Specifiers syntax**

A specifier must be preceded by a **dot**. Therefore, to apply a first specifier you need to put a **dot** after the expected text search result (which is an array of strings) and then place the specifier itself. For example, `"Expected string".from_top(0)`. The following specifiers (if any) should also be preceded by a dot: `"Expected string".from_top(0).right_bottom().move_left(10).move_up(70)`.

## An image on the screen

Aside from text, Testo Framework allows you to move the cursor to the expected image on the screen. In this case, the `destination` looks like this:

```testo
img "/path/to/img/file"
```

where `/path/to/img/file` is a path to the template of the image you expect to find on the screen. See [here](detect_img) from more information.

Moving the cursor to images is pretty much the same as moving the cursor to text. This means you can use all the same specifiers (`from_top`, `center_bottom`, `move_right` and so on).

### JS-selector

Finally, you can specify a point on the screen with a Javascript selector. A javascipt selector is a javasript-snippet (script), which must return an object with the "x" and the "y" properties.

Inside the JS-selector you may use built-in functions `find_text` and `find_img` which returns an array of all the strings and images found on the screen. The given array then could be filtered with various methods (see [here](/en/docs/js/general) for more).

In the end, specifiying the exact point in JS-selector looks a lot like specifiyng the point with text and additional selectors. You start with seraching for an array of ojects, then you specifiy the text instance you're interested in, then you move to the exact positioining and figuring out the resulting Point (an object with `x` and `y` properties). The main defference is that JS-selectors are much more flexible and allow you to perform especially tricky searching logic.

**Examples**:

```testo
# Move the mouse cursor to the center of a text "Hello world" with blue letters on the grey background.
# Will work only if there is only one instance of the "Hello world" text on the screen.
js "return find_text('Hello world').match_color('blue', 'grey').center()"

# Move the mouse cursor to the 20 pixels to the right of the right edge of the lowermost
# "Hello world" text instance on the screen
js "return find_text('Hello world').from_bottom(0).right_center().move_right(20)"

# Move the cursor to the coordinates x: 200, y: 400
js "return {x: 200, y: 400}"
```

Sometimes it is impossible to figure out the place to move the cursor to, because the expected screen event (text or image appearing) hasn't happened yet. For example, let's assume that we want to wait for exactly two "Hello world" text instances to appear on the screen, and then move the cursor in the middle between these two instances. So, naturaly, we're not okay with only one instance on the screen. Which means that the JS-selector must not return any Point until there are 2 instances present on the screen. How to act in such a situation?

In Testo-lang you can throw a built-in exception [`ContinueError`](/en/docs/js/exceptions#continueerror) inside the JS-selector which interrupts the JS-selector and instructs Testo to try JS-selector one more time a bit later (with new screenshot and new `find_text` results).

**Example**:

```testo
js """
  found = find_text('Hello world')
  // Wait until there are two instances of "Hello world" text on the screen
  if (found.size() != 2) {
    throw ContinueError()
  }

  // Coordinates of the center of the lowermost instance
  first = found.from_bottom(0).center()

  // Coordinates of the center of the uppermost instance
  second = found.from_bottom(1).center()

  result_x = (first.x + second.x) / 2
  result_y = (first.y + second.y) / 2
  // Return the coordinates of the middle Point between the 2 instances
  return {x: result_x, y: result_y}
"""
```

> `ContinueError` expection can be used only inside those JS-selectors which are related to the `mouse` actions. This exception is prohibited in the `wait` and `check` actions.

> If a `mouse` action contains a JS-selector with `ContinueError` exceptions, and the processing time of this selector exceeds the `mouse` action timeout (1 minute by default), then a timeout error will be generated.

## mouse move

Move the mouse cursor to the `destination` on the screen. When a string or JS-selector are used as the `destination`, then the `mouse move` action tries to wait for the text or image to appear on the screen, or will process the JS-selector as long (but not exceeding the `timeout_interval` time) as the `destination` won't return at least one object. If the `destination` returns two or more objects, an error is generated.

```text
mouse move <destination> [timeout timeout_interval]
```

**Arguments**:

- `destination` - Type: string, image, JS-selector or coordinates (explained above). The place to move the mouse cursor to.
- `timeout_interval` - Type: time interval or string. Timeout for the `destination` to return at least one object. Default value: `1m`. If the string type is used, the value inside the string must be convertible to a time interval. Inside the string [param referencing](param#param-referencing) is available. Default value may be changed with the `TESTO_MOUSE_MOVE_CLICK_DEFAULT_TIMEOUT` param. See [here](param#special-(reserved)-params) for more information.

## mouse click(lckick, rclick, dclick)

Perform a `click_type` mouse button click. If a `destination` is specified, then additionally a `mouse move <destination>` is implicitly called before the clicking.

```text
mouse <click_type> [destination] [timeout timeout_interval]
```

**Arguments**:

- `click_type` - Type: identifier. A mouse click type, Possible click types:. `click` (or `lclick`) - left mouse button click, `rclick` - right mouse button click, `dclick` - double left mouse button click.
- `destination` - Type: string, image, JS-selector or coordinates (explained above). The place to move the mouse cursor to.
- `timeout_interval` - Type: time interval or string. Timeout for the `destination` to return at least one object. Default value: `1m`. If the string type is used, the value inside the string must be convertible to a time interval. Inside the string [param referencing](param#param-referencing) is available. Default value may be changed with the `TESTO_MOUSE_MOVE_CLICK_DEFAULT_TIMEOUT` param. See [here](param#special-(reserved)-params) for more information.

## mouse hold

Press and hold down the mouse `button`.

```text
mouse hold <button>
```

**Arguments**:

- `<button>` - Type: identifier. Mouse button to hold down. Possible values: `lbtn`, `rbtn`.

> It is not allowed to hold more than one mouse button at a time.

> All the held mouse buttons must be released before the end of the test.

## mouse release

Release the current held down mouse button.

```text
mouse release
```

## Examples

**Basic examples**

```testo
# Move the cursor to the coordinates "Х:400, Y:0"
# Will work in any case
mouse move 400 0

# Right-click on the text "Trash bin"
# Will work only if there will be one instance of the "Trash bin" on the screen in next 10 minutes
mouse rclick "Корзина" timeout 10m
```

**Example 1**

![](/static/docs/lang/mouse_specifiers_example_1_en.png)

The objective: click on the "software" text in the red rectangle.

There are 4 instances of the text "software" on the screen, therefore a simple `mouse click "software"` won't work. But we can specify, which "software" instance we want to click. For that we could use, for example, the `from_top` specifier:

```testo
# Click the second "software" text from the top
mouse click "software".from_top(1)
```

OR we could use the `from_bottom` specifier:

```testo
# Click the third "software" text from the bottom
mouse click "software".from_bottom(2)
```

**Example 2**

![](/static/docs/lang/mouse_specifiers_example_2_en.png)

The objective: click the area in the red rectange.

To click the selected area we need to find an "anchor" to start from. In this case, the text "Preferred DNS server" could be used for such a purpose. But if we use the simple `mouse click "Preferred DNS server"`, then the cursor will be placed in the center of this text. But for us it will be more convenient to place the cursor on the right edge of the "Preferred DNS" text, because it will be closer to the desired target area. To do that we can apply the `right_center` specifier. And then, finally, we should move the cursor several pixels to the right from the right edge of the "Preferred DNS server" with the `move_right` specifier.

```testo
mouse click "Preferred DNS server".right_center().move_right(30)
```
