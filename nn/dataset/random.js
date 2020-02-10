
export function randomFloat(low, high) {
  return Math.random() * (high - low) + low
}

export function randomInt(low, high) {
  return Math.floor(Math.random() * (high - low) + low)
}

export function randomCSSFilter() {
	return `brightness(${randomFloat(0.9, 1.1)}) contrast(${randomFloat(0.9, 1.1)}) hue-rotate(${randomInt(-10, 10)}deg) saturate(${randomFloat(0.9, 1.1)})`
}

export function randomArrayElement(array) {
	return array[randomInt(0, array.length)]
}

let digits = "0123456789"
let other_symbols = "!?\"'#$%&@()[]{}<>+-*/\\.,:;^~=|_"
let english = "abcdefghijklmnopqrstuvwxyz"
let ENGLISH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
let russian = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
let RUSSIAN = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"

export let alphabet = digits + other_symbols + english + ENGLISH + russian + RUSSIAN

let char_groups = [
	digits,
	other_symbols,
	english,
	ENGLISH,
	russian,
	RUSSIAN
]

export function randomText(text) {
	let chars = text.split('')
	for (let index in chars) {
		for (let char_group of char_groups) {
			if (char_group.includes(chars[index])) {
				chars[index] = randomArrayElement(char_group)
				break
			}
		}
	}
	text = chars.join('')
	return text
}

let colors = {
	white: {
		"h": [0, 360],
		"s": [.0, .05],
		"v": [.9, 1.]
	},
	gray: {
		"h": [0, 360],
		"s": [.0, .05],
		"v": [.4, .6]
	},
	black: {
		"h": [0, 360],
		"s": [.0, .05],
		"v": [.0, .1]
	},
	red: {
		"h": [350, 370],
		"s": [.8, 1.],
		"v": [.8, 1.]
	},
	orange: {
		"h": [30, 36],
		"s": [.8, 1.],
		"v": [.8, 1.]
	},
	yellow: {
		"h": [52, 64],
		"s": [.8, 1.],
		"v": [.8, 1.]
	},
	green: {
		"h": [97, 125],
		"s": [.8, 1.],
		"v": [.8, 1.]
	},
	cyan: {
		"h": [173, 182],
		"s": [.8, 1.],
		"v": [.8, 1.]
	},
	blue: {
		"h": [220, 250],
		"s": [.8, 1.],
		"v": [.8, 1.]
	},
	purple: {
		"h": [264, 281],
		"s": [.8, 1.],
		"v": [.8, 1.]
	}
}

let colorNames = Object.keys(colors)

function hsv2rgb(hsv) {
	let {h, s, v} = hsv;

	let i = Math.floor(h * 6);
	let f = h * 6 - i;
	let p = v * (1 - s);
	let q = v * (1 - f * s);
	let t = v * (1 - (1 - f) * s);

	var r, g, b;

	switch (i % 6) {
		case 0: r = v, g = t, b = p; break;
		case 1: r = q, g = v, b = p; break;
		case 2: r = p, g = v, b = t; break;
		case 3: r = p, g = q, b = v; break;
		case 4: r = t, g = p, b = v; break;
		case 5: r = v, g = p, b = q; break;
	}

	return {
		r: Math.round(r * 255),
		g: Math.round(g * 255),
		b: Math.round(b * 255)
	};
}

export function randomColorShade(color) {
	if (!(color in colors)) {
		throw Error("unknown color: " + color)
	}
	let color_spec = colors[color]
	let h = randomInt(color_spec.h[0], color_spec.h[1]) % 360 / 360.
	let s = randomFloat(color_spec.s[0], color_spec.s[1])
	let v = randomFloat(color_spec.v[0], color_spec.v[1])
	let {r, g, b} = hsv2rgb({h, s, v})
	return `RGB(${r}, ${g}, ${b})`
}

export function randomColor() {
	return randomArrayElement(colorNames)
}

export function randomColors() {
	let fg = randomColor()
	while (true) {
		let bg = randomColor()
		if (fg != bg) {
			return {fg, bg}
		}
	}
}
