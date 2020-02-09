
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
