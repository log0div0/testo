# Class Point

The `Point` class contains the information about a point on the virtual machine screen.

## Methods

### move_up(N)

Returns a new Point with the `y` coordinate reduced by N pixels compared to the current object.

**Arguments**:

- `N <integer>` - Number of pixels for the new Point to be "higher" than the current Point object.

**Return value** - a `Point` object with the new coordinates.

### move_down(N)

Returns a new Point with the `y` coordinate increased by N pixels compared to the current object.

**Arguments**:

- `N <integer>` - Number of pixels for the new Point to be "lower" than the current Point object.

**Return value** - a `Point` object with the new coordinates.

### move_right(N)

Returns a new Point with the `x` coordinate increased by N pixels compared to the current object.

**Arguments**:

- `N <integer>` - Number of pixels for the new Point to be to the right from the current Point object.

**Return value** - a `Point` object with the new coordinates.

### move_left(N)

Returns a new Point with the `x` coordinate decreased by N pixels compared to the current object.

**Arguments**:

- `N <integer>` - Number of pixels for the new Point to be to the left from the current Point object.

**Return value** - a `Point` object with the new coordinates.

## Properties

### `x <integer>`

The X coordinate value (more `x` means more to the right).

### `y <integer>`

The Y coordinate value (more `y` means more to the bottom).
