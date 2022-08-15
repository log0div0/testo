# Class ImgTensor

Class `ImgTensor` contains the information about images on the screen. Could be obtained with a [`find_img`](global_funcs#find_img()) global function call.

## Methods

### from_top(index)

Select the image with the specified index from the array of images sorted from top to bottom.

**Arguments**:

- `index <integer>` - the index of the image to be selected. With the `index == 0` the "uppermost" image on the screen will be selected.

**Return value** - new object of `ImgTensor`, containing just one selected image.

> Images are sorted by the Y coordinate of the top edge of the images.

### from_bottom(index)

Select the image with the specified index from the array of images sorted from bottom to top.

**Arguments**:

- `index <integer>` - the index of the image to be selected. With the `index == 0` the "lowermost" image on the screen will be selected.

**Return value** - new object of `ImgTensor`, containing just one selected image.

> Images are sorted by the Y coordinate of the bottom edge of the images.

### from_left(index)

Select the image with the specified index from the array of images sorted from left to right.

**Arguments**:

- `index <integer>` - the index of the image to be selected. With the `index == 0` the most left image on the screen will be selected.

**Return value** - new object of `ImgTensor`, containing just one selected image.

> Images are sorted by the X coordinate of the left edge of the images.

### from_right(index)

Select the image with the specified index from the array of images sorted from right to left.

**Arguments**:

- `index <integer>` - the index of the image to be selected. With the `index == 0` the most right image on the screen will be selected.

**Return value** - new object of `ImgTensor`, containing just one selected image.

> Images are sorted by the X coordinate of the right edge of the images.

### center()

Return a Point with the coordinates of the geometric center of the image.

**Arguments**: no

**Return value** - new object `Point`, containing the coordinates of the geometic center of the image.

> This method is accessible only when the current `ImgTensor` contains exactly one image. Otherwise an error is generated.

### left_bottom()

Return a Point with the coordinates of the bottom left corner of the image.

**Arguments**: no

**Return value** - new object `Point`, containing the coordinates of the bottom left corner of the image.

> This method is accessible only when the current `ImgTensor` contains exactly one image. Otherwise an error is generated.

### left_center()

Return a Point with the coordinates of the center of the left edge of the image.

**Arguments**: no

**Return value** - new object `Point`, containing the coordinates of the center of the left edge of the image.

> This method is accessible only when the current `ImgTensor` contains exactly one image. Otherwise an error is generated.

### left_top()

Return a Point with the coordinates of the top left corner of the image.

**Arguments**: no

**Return value** - new object `Point`, containing the coordinates of the top left corner of the image.

> This method is accessible only when the current `ImgTensor` contains exactly one image. Otherwise an error is generated.

### center_bottom()

Return a Point with the coordinates of the center of the left edge of the image.

**Arguments**: no

**Return value** - new object `Point`, containing the coordinates of the center of the left edge of the image.

> This method is accessible only when the current `ImgTensor` contains exactly one image. Otherwise an error is generated.

### center_top()

Return a Point with the coordinates of the center of the top edge of the image.

**Arguments**: no

**Return value** - new object `Point`, containing the coordinates of the center of the top edge of the image.

> This method is accessible only when the current `ImgTensor` contains exactly one image. Otherwise an error is generated.

### right_bottom()

Return a Point with the coordinates of the bottom right corner of the image.

**Arguments**: no

**Return value** - new object `Point`, containing the coordinates of the bottom right corner of the image.

> This method is accessible only when the current `ImgTensor` contains exactly one image. Otherwise an error is generated.

### right_center()

Return a Point with the coordinates of the center of the right edge of the image.

**Arguments**: no

**Return value** - new object `Point`, containing the coordinates of the center of the right edge of the image.

> This method is accessible only when the current `ImgTensor` contains exactly one image. Otherwise an error is generated.

### right_top()

Return a Point with the coordinates of the top right corner of the image.

**Arguments**: no

**Return value** - new object `Point`, containing the coordinates of the top right corner of the image.

> This method is accessible only when the current `ImgTensor` contains exactly one image. Otherwise an error is generated.

## Properties

### `x <integer>`

The X coordinate value (the bigger `x` means more towards the right).

> This property is accessible only when the current `ImgTensor` contains exactly one image. Otherwise an error is generated.

### `y <integer>`

The Y coordinate value (the bigger `y` means more towards the bottom)

> This property is accessible only when the current `ImgTensor` contains exactly one image. Otherwise an error is generated.
