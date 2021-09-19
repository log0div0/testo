
#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#ifdef DELETE
#undef DELETE // fuck windows
#endif

enum class KeyboardButton {
	ESC,
	ONE,
	TWO,
	THREE,
	FOUR,
	FIVE,
	SIX,
	SEVEN,
	EIGHT,
	NINE,
	ZERO,
	MINUS,
	EQUALSIGN,
	BACKSPACE,
	TAB,
	Q,
	W,
	E,
	R,
	T,
	Y,
	U,
	I,
	O,
	P,
	LEFTBRACE,
	RIGHTBRACE,
	ENTER,
	LEFTCTRL,
	A,
	S,
	D,
	F,
	G,
	H,
	J,
	K,
	L,
	SEMICOLON,
	APOSTROPHE,
	GRAVE,
	LEFTSHIFT,
	BACKSLASH,
	Z,
	X,
	C,
	V,
	B,
	N,
	M,
	COMMA,
	DOT,
	SLASH,
	RIGHTSHIFT,
	LEFTALT,
	SPACE,
	CAPSLOCK,
	F1,
	F2,
	F3,
	F4,
	F5,
	F6,
	F7,
	F8,
	F9,
	F10,
	F11,
	F12,
	NUMLOCK,
	KP_0,
	KP_1,
	KP_2,
	KP_3,
	KP_4,
	KP_5,
	KP_6,
	KP_7,
	KP_8,
	KP_9,
	KP_PLUS,
	KP_MINUS,
	KP_SLASH,
	KP_ASTERISK,
	KP_ENTER,
	KP_DOT,
	SCROLLLOCK,
	RIGHTCTRL,
	RIGHTALT,
	HOME,
	UP,
	PAGEUP,
	LEFT,
	RIGHT,
	END,
	DOWN,
	PAGEDOWN,
	INSERT,
	DELETE,
	LEFTMETA,
	RIGHTMETA,
	SCROLLUP,
	SCROLLDOWN,
};

std::string ToString(KeyboardButton button);
KeyboardButton ToKeyboardButton(const std::string& button);

struct CharDesc {
	KeyboardButton button;
	bool hold_shift = false;
};
using CharMap = std::unordered_map<char32_t, CharDesc>;

enum class KeyboardAction {
	Hold,
	Release
};

struct KeyboardCommand {
	KeyboardAction action;
	KeyboardButton button;
};

bool operator==(const KeyboardCommand& a, const KeyboardCommand& b);

struct KeyboardLayout;

struct TextChunk {
	const KeyboardLayout* layout;
	std::string text;
};

bool operator==(const TextChunk& a, const TextChunk& b);

struct KeyboardLayout {
	static const KeyboardLayout US;
	static const KeyboardLayout RU;
	static std::vector<TextChunk> split_text_by_layout(const std::string& text);
	static std::vector<TextChunk> split_text_by_layout(const std::u32string& text);

	std::vector<KeyboardCommand> type(const std::string& text) const;
	std::vector<KeyboardCommand> type(const std::u32string& text) const;
	const CharDesc& find_char_desc(char32_t ch) const;
	bool can_type(char32_t ch) const;

	std::string name;
	CharMap char_map;
};
