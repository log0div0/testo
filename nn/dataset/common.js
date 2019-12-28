
import React from 'react'
import fs from 'fs'
import path from 'path'

export function random(low, high) {
  return Math.random() * (high - low) + low
}

export function randomInt(low, high) {
  return Math.floor(Math.random() * (high - low) + low)
}

export function randomCSSFilter() {
	return `brightness(${random(0.9, 1.1)}) contrast(${random(0.9, 1.1)}) hue-rotate(${randomInt(-10, 10)}deg) saturate(${random(0.9, 1.1)})`
}

export function randomArrayElement(array) {
	return array[randomInt(0, array.length)]
}

let char_groups = [
	'0123456789',
	'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
	'abcdefghijklmnopqrstuvwxyz',
	'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ',
	'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
]

let common = "0123456789!?\"'#$%&@()[]{}<>+-*/\\.,:;^~=|_"
let english = "abcdefghijklmnopqrstuvwxyz"
let ENGLISH = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
let russian = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
let RUSSIAN = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
export let alphabet = common + english + ENGLISH + russian + RUSSIAN

export function Background(props) {
	let divStyle = {
		display: 'inline-block',
		position: 'relative'
	}
	let imgStyle = {
		filter: randomCSSFilter(),
		verticalAlign: 'bottom'
	}

	return (
		<div style={divStyle}>
			<img style={imgStyle} src={props.src}/>
			{props.children}
		</div>
	)
}

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

function _textToChars(text, Char, props) {
	text = randomText(text)
	let chars = []
	for (let index in text) {
		chars.push(<Char key={index} {...props}>{text[index]}</Char>)
	}
	return chars
}

function textToChars(children, Char, props) {
	if (Array.isArray(children)) {
		return children.map(child => {
			if ((typeof child == "string") || (child instanceof String)) {
				return _textToChars(child, Char, props)
			} else {
				return child
			}
		})
	} else {
		if ((typeof children == "string") || (children instanceof String)) {
			return _textToChars(children, Char, props)
		} else {
			return children
		}
	}
}

function defaultStyle(props) {
	let {style, ...other} = props
	if (!style) {
		style = {}
	}
	return {style, ...other}
}

function positionStyle(props) {
	let {style, ...other} = props
	if (props.left) {
		style.position = 'absolute'
		style.left = props.left
	}
	if (props.right) {
		style.position = 'absolute'
		style.right = props.right
	}
	if (props.top) {
		style.position = 'absolute'
		style.top = props.top
	}
	if (props.bottom) {
		style.position = 'absolute'
		style.bottom = props.bottom
	}
	if (props.x && props.y) {
		style.position = 'absolute'
		style.left = props.x
		style.top = props.y
		style.transform = 'translate(-50%,-50%)'
	} else if (props.x) {
		style.position = 'absolute'
		style.left = props.x
		style.transform = 'translate(-50%,0)'
	} else if (props.y) {
		style.position = 'absolute'
		style.top = props.y
		style.transform = 'translate(0,-50%)'
	}
	return {style, ...other}
}

function alignmentStyle(props) {
	let {style, ...other} = props
	if (props.left) {
		style.display = 'flex'
		style.alignItems = 'flex-start'
	}
	if (props.right) {
		style.display = 'flex'
		style.alignItems = 'flex-end'
	}
	if (props.top) {
		style.display = 'flex'
		style.justifyContent = 'flex-start'
	}
	if (props.bottom) {
		style.display = 'flex'
		style.justifyContent = 'flex-end'
	}
	if (props.x && props.y) {
		style.display = 'flex'
		style.alignItems = 'center'
		style.justifyContent = 'center'
	} else if (props.x) {
		style.display = 'flex'
		style.alignItems = 'center'
	} else if (props.y) {
		style.display = 'flex'
		style.justifyContent = 'center'
	}
	return {style, ...other}
}

function Char(props) {
	return <span className='char'>{props.children}</span>
}

export function Text(props) {
	let {children, style, className} = props
	return <span style={style} className={className}>{textToChars(children, Char, {})}</span>
}

export function TextLine(props) {
	let {children, ...other} = positionStyle(defaultStyle(props))
	other.style.filter = randomCSSFilter()
	other.style.whiteSpace = 'nowrap'
	other.className = 'textline'

	return <Text {...other}>{children}</Text>
}

function ConsoleChar(props) {
	let style = {
		color: 'transparent',
		width: '8px',
		height: '16px'
	}
	if (props.children != ' ') {
		let index = alphabet.indexOf(props.children)
		if (index < 0) {
			throw Error("Unknown char: " + props.children)
		}

		let console_fonts_dir = path.join(__dirname, 'fonts', 'console')
		let img_path = `${console_fonts_dir}/${props.font}/${index}.png`
		let img_data = fs.readFileSync(img_path)
		let img = "data:image/png;base64," + img_data.toString('base64')

		style.backgroundColor = props.color
		style.WebkitMaskImage = `url(${img})`
	}
	return <span className='char' style={style}>{props.children}</span>
}

export function ConsoleText(props) {
	let {children, color, font, style, className} = defaultStyle(props)
	style.display = 'flex'
	return <span style={style} className={className}>{textToChars(children, ConsoleChar, {color, font})}</span>
}

export function ConsoleTextLine(props) {
	let {children, ...other} = positionStyle(defaultStyle(props))
	other.style.filter = randomCSSFilter()
	other.className = 'textline'

	return <ConsoleText {...other}>{children}</ConsoleText>
}

export function Column(props) {
	let {style, children} = alignmentStyle(positionStyle(defaultStyle(props)))
	style.flexDirection = 'column'
	return (
		<div style={style}>{children}</div>
	)
}

export function Row(props) {
	let {style, children} = alignmentStyle(positionStyle(defaultStyle(props)))
	style.flexDirection = 'row'
	return (
		<div style={style}>{children}</div>
	)
}
