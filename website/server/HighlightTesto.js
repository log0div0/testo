
export default function(hljs) {
	let STRING = {
		className: 'string',
		variants: [
			{
				begin: /"""/,
				end: /"""/
			},
			{
				begin: /"/,
				end: /"/
			}
		],
		contains: [
			{
				className: 'literal',
				variants: [
					{ begin: '\\\\"""' },
					{ begin: '\\\\"' },
					{ begin: '\\\\n' }
				]
			},
			{
				className: 'subst',
				begin: "\\${",
				end: "}"
			}
		]
	}
	let NUMBER = {
		className: 'number',
		begin: /\b\d+(Kb|Mb|Gb|s|m|h)?\b/
	}
	let KEYWORDS = {
		keyword: 'for if else continue break machine',
		literal: 'true false'
	}
	return {
		name: 'Testo Lang',
		keywords: KEYWORDS,
		contains: [
			STRING,
			NUMBER,
			hljs.C_BLOCK_COMMENT_MODE,
			hljs.HASH_COMMENT_MODE,
			{
				className: 'function',
				beginKeywords: 'machine test network flash',
				end: /{/,
				excludeEnd: true,
				contains: [
					hljs.UNDERSCORE_TITLE_MODE
				]
			},
			{
				className: 'function',
				keywords: 'param',
				begin: "param\\s+",
				contains: [
					hljs.UNDERSCORE_TITLE_MODE
				]
			},
			{
				className: 'attribute',
				begin: /\b(type|wait|press|plug|unplug|start|stop|exec|copyto|copyfrom|shutdown|print|abort|mouse|sleep)\b/
			},
			{
				className: 'strong',
				begin: /\b(EQUAL|LESS|GREATER|STREQUAL|STRGREATER|STRLESS|IN|RANGE)\b/
			},
			{
				className: 'attr',
				begin: /\b(NOT|AND|OR|check)\b/
			},
		]
	}
}
