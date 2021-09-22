
#include "Keyboard.hpp"
#include <map>
#include <locale>
#include <codecvt>
#include <algorithm>
#include <list>
#include <cassert>
#include <iostream>

const std::vector<std::string> kb_to_str = {
	"ESC",
	"ONE",
	"TWO",
	"THREE",
	"FOUR",
	"FIVE",
	"SIX",
	"SEVEN",
	"EIGHT",
	"NINE",
	"ZERO",
	"MINUS",
	"EQUALSIGN",
	"BACKSPACE",
	"TAB",
	"Q",
	"W",
	"E",
	"R",
	"T",
	"Y",
	"U",
	"I",
	"O",
	"P",
	"LEFTBRACE",
	"RIGHTBRACE",
	"ENTER",
	"LEFTCTRL",
	"A",
	"S",
	"D",
	"F",
	"G",
	"H",
	"J",
	"K",
	"L",
	"SEMICOLON",
	"APOSTROPHE",
	"GRAVE",
	"LEFTSHIFT",
	"BACKSLASH",
	"Z",
	"X",
	"C",
	"V",
	"B",
	"N",
	"M",
	"COMMA",
	"DOT",
	"SLASH",
	"RIGHTSHIFT",
	"LEFTALT",
	"SPACE",
	"CAPSLOCK",
	"F1",
	"F2",
	"F3",
	"F4",
	"F5",
	"F6",
	"F7",
	"F8",
	"F9",
	"F10",
	"F11",
	"F12",
	"NUMLOCK",
	"KP_0",
	"KP_1",
	"KP_2",
	"KP_3",
	"KP_4",
	"KP_5",
	"KP_6",
	"KP_7",
	"KP_8",
	"KP_9",
	"KP_PLUS",
	"KP_MINUS",
	"KP_SLASH",
	"KP_ASTERISK",
	"KP_ENTER",
	"KP_DOT",
	"SCROLLLOCK",
	"RIGHTCTRL",
	"RIGHTALT",
	"HOME",
	"UP",
	"PAGEUP",
	"LEFT",
	"RIGHT",
	"END",
	"DOWN",
	"PAGEDOWN",
	"INSERT",
	"DELETE",
	"LEFTMETA",
	"RIGHTMETA",
	"SCROLLUP",
	"SCROLLDOWN",
};

const std::map<std::string, KeyboardButton> str_to_kb = {
	{"ESC", KeyboardButton::ESC},
	{"ONE", KeyboardButton::ONE},
	{"TWO", KeyboardButton::TWO},
	{"THREE", KeyboardButton::THREE},
	{"FOUR", KeyboardButton::FOUR},
	{"FIVE", KeyboardButton::FIVE},
	{"SIX", KeyboardButton::SIX},
	{"SEVEN", KeyboardButton::SEVEN},
	{"EIGHT", KeyboardButton::EIGHT},
	{"NINE", KeyboardButton::NINE},
	{"ZERO", KeyboardButton::ZERO},
	{"MINUS", KeyboardButton::MINUS},
	{"EQUALSIGN", KeyboardButton::EQUALSIGN},
	{"BACKSPACE", KeyboardButton::BACKSPACE},
	{"TAB", KeyboardButton::TAB},
	{"Q", KeyboardButton::Q},
	{"W", KeyboardButton::W},
	{"E", KeyboardButton::E},
	{"R", KeyboardButton::R},
	{"T", KeyboardButton::T},
	{"Y", KeyboardButton::Y},
	{"U", KeyboardButton::U},
	{"I", KeyboardButton::I},
	{"O", KeyboardButton::O},
	{"P", KeyboardButton::P},
	{"LEFTBRACE", KeyboardButton::LEFTBRACE},
	{"RIGHTBRACE", KeyboardButton::RIGHTBRACE},
	{"ENTER", KeyboardButton::ENTER},
	{"LEFTCTRL", KeyboardButton::LEFTCTRL},
	{"A", KeyboardButton::A},
	{"S", KeyboardButton::S},
	{"D", KeyboardButton::D},
	{"F", KeyboardButton::F},
	{"G", KeyboardButton::G},
	{"H", KeyboardButton::H},
	{"J", KeyboardButton::J},
	{"K", KeyboardButton::K},
	{"L", KeyboardButton::L},
	{"SEMICOLON", KeyboardButton::SEMICOLON},
	{"APOSTROPHE", KeyboardButton::APOSTROPHE},
	{"GRAVE", KeyboardButton::GRAVE},
	{"LEFTSHIFT", KeyboardButton::LEFTSHIFT},
	{"BACKSLASH", KeyboardButton::BACKSLASH},
	{"Z", KeyboardButton::Z},
	{"X", KeyboardButton::X},
	{"C", KeyboardButton::C},
	{"V", KeyboardButton::V},
	{"B", KeyboardButton::B},
	{"N", KeyboardButton::N},
	{"M", KeyboardButton::M},
	{"COMMA", KeyboardButton::COMMA},
	{"DOT", KeyboardButton::DOT},
	{"SLASH", KeyboardButton::SLASH},
	{"RIGHTSHIFT", KeyboardButton::RIGHTSHIFT},
	{"LEFTALT", KeyboardButton::LEFTALT},
	{"SPACE", KeyboardButton::SPACE},
	{"CAPSLOCK", KeyboardButton::CAPSLOCK},
	{"F1", KeyboardButton::F1},
	{"F2", KeyboardButton::F2},
	{"F3", KeyboardButton::F3},
	{"F4", KeyboardButton::F4},
	{"F5", KeyboardButton::F5},
	{"F6", KeyboardButton::F6},
	{"F7", KeyboardButton::F7},
	{"F8", KeyboardButton::F8},
	{"F9", KeyboardButton::F9},
	{"F10", KeyboardButton::F10},
	{"F11", KeyboardButton::F11},
	{"F12", KeyboardButton::F12},
	{"NUMLOCK", KeyboardButton::NUMLOCK},
	{"KP_0", KeyboardButton::KP_0},
	{"KP_1", KeyboardButton::KP_1},
	{"KP_2", KeyboardButton::KP_2},
	{"KP_3", KeyboardButton::KP_3},
	{"KP_4", KeyboardButton::KP_4},
	{"KP_5", KeyboardButton::KP_5},
	{"KP_6", KeyboardButton::KP_6},
	{"KP_7", KeyboardButton::KP_7},
	{"KP_8", KeyboardButton::KP_8},
	{"KP_9", KeyboardButton::KP_9},
	{"KP_PLUS", KeyboardButton::KP_PLUS},
	{"KP_MINUS", KeyboardButton::KP_MINUS},
	{"KP_SLASH", KeyboardButton::KP_SLASH},
	{"KP_ASTERISK", KeyboardButton::KP_ASTERISK},
	{"KP_ENTER", KeyboardButton::KP_ENTER},
	{"KP_DOT", KeyboardButton::KP_DOT},
	{"SCROLLLOCK", KeyboardButton::SCROLLLOCK},
	{"RIGHTCTRL", KeyboardButton::RIGHTCTRL},
	{"RIGHTALT", KeyboardButton::RIGHTALT},
	{"HOME", KeyboardButton::HOME},
	{"UP", KeyboardButton::UP},
	{"PAGEUP", KeyboardButton::PAGEUP},
	{"LEFT", KeyboardButton::LEFT},
	{"RIGHT", KeyboardButton::RIGHT},
	{"END", KeyboardButton::END},
	{"DOWN", KeyboardButton::DOWN},
	{"PAGEDOWN", KeyboardButton::PAGEDOWN},
	{"INSERT", KeyboardButton::INSERT},
	{"DELETE", KeyboardButton::DELETE},
	{"LEFTMETA", KeyboardButton::LEFTMETA},
	{"RIGHTMETA", KeyboardButton::RIGHTMETA},
	{"SCROLLUP", KeyboardButton::SCROLLUP},
	{"SCROLLDOWN", KeyboardButton::SCROLLDOWN},
};

std::string ToString(KeyboardButton button) {
	return kb_to_str.at((int)button);
}

KeyboardButton ToKeyboardButton(const std::string& button) {
	std::string button_str = button;
	std::transform(button_str.begin(), button_str.end(), button_str.begin(), ::toupper);
	auto it = str_to_kb.find(button_str);
	if (it == str_to_kb.end()) {
		throw std::runtime_error("Error: unknown key: " + button);
	}
	return it->second;
}

const KeyboardLayout KeyboardLayout::US = { "US", U"fwz", {
	{U'0', {KeyboardButton::ZERO}},
	{U'1', {KeyboardButton::ONE}},
	{U'2', {KeyboardButton::TWO}},
	{U'3', {KeyboardButton::THREE}},
	{U'4', {KeyboardButton::FOUR}},
	{U'5', {KeyboardButton::FIVE}},
	{U'6', {KeyboardButton::SIX}},
	{U'7', {KeyboardButton::SEVEN}},
	{U'8', {KeyboardButton::EIGHT}},
	{U'9', {KeyboardButton::NINE}},

	{U'!', {KeyboardButton::ONE, true}},
	{U'@', {KeyboardButton::TWO, true}},
	{U'#', {KeyboardButton::THREE, true}},
	{U'$', {KeyboardButton::FOUR, true}},
	{U'%', {KeyboardButton::FIVE, true}},
	{U'^', {KeyboardButton::SIX, true}},
	{U'&', {KeyboardButton::SEVEN, true}},
	{U'*', {KeyboardButton::EIGHT, true}},
	{U'(', {KeyboardButton::NINE, true}},
	{U')', {KeyboardButton::ZERO, true}},

	{U'`', {KeyboardButton::GRAVE}},
	{U'~', {KeyboardButton::GRAVE, true}},
	{U'-', {KeyboardButton::MINUS}},
	{U'_', {KeyboardButton::MINUS, true}},
	{U'=', {KeyboardButton::EQUALSIGN}},
	{U'+', {KeyboardButton::EQUALSIGN, true}},
	{U'\'', {KeyboardButton::APOSTROPHE}},
	{U'\"', {KeyboardButton::APOSTROPHE, true}},
	{U'\\', {KeyboardButton::BACKSLASH}},
	{U'|', {KeyboardButton::BACKSLASH, true}},
	{U',', {KeyboardButton::COMMA}},
	{U'<', {KeyboardButton::COMMA, true}},
	{U'.', {KeyboardButton::DOT}},
	{U'>', {KeyboardButton::DOT, true}},
	{U'/', {KeyboardButton::SLASH}},
	{U'?', {KeyboardButton::SLASH, true}},
	{U';', {KeyboardButton::SEMICOLON}},
	{U':', {KeyboardButton::SEMICOLON, true}},
	{U'[', {KeyboardButton::LEFTBRACE}},
	{U'{', {KeyboardButton::LEFTBRACE, true}},
	{U']', {KeyboardButton::RIGHTBRACE}},
	{U'}', {KeyboardButton::RIGHTBRACE, true}},

	{U'\n', {KeyboardButton::ENTER}},
	{U'\t', {KeyboardButton::TAB}},
	{U' ', {KeyboardButton::SPACE}},

	{U'a', {KeyboardButton::A}},
	{U'b', {KeyboardButton::B}},
	{U'c', {KeyboardButton::C}},
	{U'd', {KeyboardButton::D}},
	{U'e', {KeyboardButton::E}},
	{U'f', {KeyboardButton::F}},
	{U'g', {KeyboardButton::G}},
	{U'h', {KeyboardButton::H}},
	{U'i', {KeyboardButton::I}},
	{U'j', {KeyboardButton::J}},
	{U'k', {KeyboardButton::K}},
	{U'l', {KeyboardButton::L}},
	{U'm', {KeyboardButton::M}},
	{U'n', {KeyboardButton::N}},
	{U'o', {KeyboardButton::O}},
	{U'p', {KeyboardButton::P}},
	{U'q', {KeyboardButton::Q}},
	{U'r', {KeyboardButton::R}},
	{U's', {KeyboardButton::S}},
	{U't', {KeyboardButton::T}},
	{U'u', {KeyboardButton::U}},
	{U'v', {KeyboardButton::V}},
	{U'w', {KeyboardButton::W}},
	{U'x', {KeyboardButton::X}},
	{U'y', {KeyboardButton::Y}},
	{U'z', {KeyboardButton::Z}},

	{U'A', {KeyboardButton::A, true}},
	{U'B', {KeyboardButton::B, true}},
	{U'C', {KeyboardButton::C, true}},
	{U'D', {KeyboardButton::D, true}},
	{U'E', {KeyboardButton::E, true}},
	{U'F', {KeyboardButton::F, true}},
	{U'G', {KeyboardButton::G, true}},
	{U'H', {KeyboardButton::H, true}},
	{U'I', {KeyboardButton::I, true}},
	{U'J', {KeyboardButton::J, true}},
	{U'K', {KeyboardButton::K, true}},
	{U'L', {KeyboardButton::L, true}},
	{U'M', {KeyboardButton::M, true}},
	{U'N', {KeyboardButton::N, true}},
	{U'O', {KeyboardButton::O, true}},
	{U'P', {KeyboardButton::P, true}},
	{U'Q', {KeyboardButton::Q, true}},
	{U'R', {KeyboardButton::R, true}},
	{U'S', {KeyboardButton::S, true}},
	{U'T', {KeyboardButton::T, true}},
	{U'U', {KeyboardButton::U, true}},
	{U'V', {KeyboardButton::V, true}},
	{U'W', {KeyboardButton::W, true}},
	{U'X', {KeyboardButton::X, true}},
	{U'Y', {KeyboardButton::Y, true}},
	{U'Z', {KeyboardButton::Z, true}},
}};

const KeyboardLayout KeyboardLayout::RU = { "RU", U"яфц", {
	{U'0', {KeyboardButton::ZERO}},
	{U'1', {KeyboardButton::ONE}},
	{U'2', {KeyboardButton::TWO}},
	{U'3', {KeyboardButton::THREE}},
	{U'4', {KeyboardButton::FOUR}},
	{U'5', {KeyboardButton::FIVE}},
	{U'6', {KeyboardButton::SIX}},
	{U'7', {KeyboardButton::SEVEN}},
	{U'8', {KeyboardButton::EIGHT}},
	{U'9', {KeyboardButton::NINE}},

	{U'!', {KeyboardButton::ONE, true}},
	{U'"', {KeyboardButton::TWO, true}},
	{U'№', {KeyboardButton::THREE, true}},
	{U';', {KeyboardButton::FOUR, true}},
	{U'%', {KeyboardButton::FIVE, true}},
	{U':', {KeyboardButton::SIX, true}},
	{U'?', {KeyboardButton::SEVEN, true}},
	{U'*', {KeyboardButton::EIGHT, true}},
	{U'(', {KeyboardButton::NINE, true}},
	{U')', {KeyboardButton::ZERO, true}},

	{U'-', {KeyboardButton::MINUS}},
	{U'_', {KeyboardButton::MINUS, true}},
	{U'=', {KeyboardButton::EQUALSIGN}},
	{U'+', {KeyboardButton::EQUALSIGN, true}},
	{U'\\', {KeyboardButton::BACKSLASH}},
	{U'/', {KeyboardButton::BACKSLASH, true}},
	{U'.', {KeyboardButton::SLASH}},
	{U',', {KeyboardButton::SLASH, true}},

	{U'\n', {KeyboardButton::ENTER}},
	{U'\t', {KeyboardButton::TAB}},
	{U' ', {KeyboardButton::SPACE}},

	{U'а', {KeyboardButton::F}},
	{U'б', {KeyboardButton::COMMA}},
	{U'в', {KeyboardButton::D}},
	{U'г', {KeyboardButton::U}},
	{U'д', {KeyboardButton::L}},
	{U'е', {KeyboardButton::T}},
	{U'ё', {KeyboardButton::GRAVE}},
	{U'ж', {KeyboardButton::SEMICOLON}},
	{U'з', {KeyboardButton::P}},
	{U'и', {KeyboardButton::B}},
	{U'й', {KeyboardButton::Q}},
	{U'к', {KeyboardButton::R}},
	{U'л', {KeyboardButton::K}},
	{U'м', {KeyboardButton::V}},
	{U'н', {KeyboardButton::Y}},
	{U'о', {KeyboardButton::J}},
	{U'п', {KeyboardButton::G}},
	{U'р', {KeyboardButton::H}},
	{U'с', {KeyboardButton::C}},
	{U'т', {KeyboardButton::N}},
	{U'у', {KeyboardButton::E}},
	{U'ф', {KeyboardButton::A}},
	{U'х', {KeyboardButton::LEFTBRACE}},
	{U'ц', {KeyboardButton::W}},
	{U'ч', {KeyboardButton::X}},
	{U'ш', {KeyboardButton::I}},
	{U'щ', {KeyboardButton::O}},
	{U'ъ', {KeyboardButton::RIGHTBRACE}},
	{U'ы', {KeyboardButton::S}},
	{U'ь', {KeyboardButton::M}},
	{U'э', {KeyboardButton::APOSTROPHE}},
	{U'ю', {KeyboardButton::DOT}},
	{U'я', {KeyboardButton::Z}},

	{U'А', {KeyboardButton::F, true}},
	{U'Б', {KeyboardButton::COMMA, true}},
	{U'В', {KeyboardButton::D, true}},
	{U'Г', {KeyboardButton::U, true}},
	{U'Д', {KeyboardButton::L, true}},
	{U'Е', {KeyboardButton::T, true}},
	{U'Ё', {KeyboardButton::GRAVE, true}},
	{U'Ж', {KeyboardButton::SEMICOLON, true}},
	{U'З', {KeyboardButton::P, true}},
	{U'И', {KeyboardButton::B, true}},
	{U'Й', {KeyboardButton::Q, true}},
	{U'К', {KeyboardButton::R, true}},
	{U'Л', {KeyboardButton::K, true}},
	{U'М', {KeyboardButton::V, true}},
	{U'Н', {KeyboardButton::Y, true}},
	{U'О', {KeyboardButton::J, true}},
	{U'П', {KeyboardButton::G, true}},
	{U'Р', {KeyboardButton::H, true}},
	{U'С', {KeyboardButton::C, true}},
	{U'Т', {KeyboardButton::N, true}},
	{U'У', {KeyboardButton::E, true}},
	{U'Ф', {KeyboardButton::A, true}},
	{U'Х', {KeyboardButton::LEFTBRACE, true}},
	{U'Ц', {KeyboardButton::W, true}},
	{U'Ч', {KeyboardButton::X, true}},
	{U'Ш', {KeyboardButton::I, true}},
	{U'Щ', {KeyboardButton::O, true}},
	{U'Ъ', {KeyboardButton::RIGHTBRACE, true}},
	{U'Ы', {KeyboardButton::S, true}},
	{U'Ь', {KeyboardButton::M, true}},
	{U'Э', {KeyboardButton::APOSTROPHE, true}},
	{U'Ю', {KeyboardButton::DOT, true}},
	{U'Я', {KeyboardButton::Z, true}},
}};

bool operator==(const KeyboardCommand& a, const KeyboardCommand& b) {
	return (a.action == b.action) && (a.button == b.button);
}

bool operator==(const TypingPlan& a, const TypingPlan& b) {
	return (a.layout == b.layout)
		&& (a.prefix == b.prefix)
		&& (a.core == b.core)
		&& (a.postfix == b.postfix)
		&& (a.tail == b.tail)
	;
}

std::ostream& operator<<(std::ostream& stream, const TypingPlan& x) {
	return stream
		<< "{"
		<< x.layout->name << ", "
		<< x.prefix << ", "
		<< x.core << ", "
		<< x.postfix << ", "
		<< x.tail
		<< "}"
	;
}

std::string TypingPlan::what_to_search() const {
	return prefix + core + postfix;
}

std::vector<KeyboardCommand> TypingPlan::start_typing() const {
	return layout->type(core + postfix);
}

std::vector<KeyboardCommand> TypingPlan::rollback() const {
	return layout->clear(core + postfix);
}

std::vector<KeyboardCommand> TypingPlan::finish_typing() const {
	auto a = layout->clear(postfix);
	auto b = layout->type(tail);
	a.insert(a.end(), b.begin(), b.end());
	return a;
}

std::vector<KeyboardCommand> TypingPlan::just_type_final_text() const {
	return layout->type(core + tail);
}

static std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;

std::list<const KeyboardLayout*> GetAvailableLayouts() {
	return {&KeyboardLayout::US, &KeyboardLayout::RU};
}

static std::u32string common_chars = U"0123456789!%*()-_=+\\ \n\t";

bool IsCommonChar(char32_t ch) {
	for (char32_t x: common_chars) {
		if (ch == x) {
			return true;
		}
	}
	return false;
}

bool IsWhiteSpace(char32_t ch) {
	return (ch == U' ') || (ch == U'\t') || (ch == U'\n');
}

size_t FindNextNotCommonChar(const std::u32string& text, size_t i) {
	size_t j = i + 1;
	while ((j < text.size()) && IsCommonChar(text[j])) {
		++j;
	}
	return j;
}

size_t FindRightBound(const std::u32string& text, size_t i, size_t end) {
	size_t j = i + 1;
	while ((j < end) && !IsWhiteSpace(text[j])) {
		++j;
	}
	return j;
}

size_t FindLeftBound(const std::u32string& text, size_t i) {
	int64_t j = int64_t(i) - 1;
	while ((j >= 0) && !IsWhiteSpace(text[j])) {
		--j;
	}
	return j + 1;
}

std::tuple<size_t, const KeyboardLayout*> ChooseLayout(const std::u32string& text, size_t i) {
	std::list<const KeyboardLayout*> layouts = GetAvailableLayouts();
	const KeyboardLayout* layout = nullptr;
	while (i < text.size()) {
		for (auto it = layouts.begin(); it != layouts.end();) {
			if (!(*it)->can_type(text[i])) {
				layouts.erase(it++);
			} else {
				++it;
			}
		}
		if (!layouts.size()) {
			break;
		} else {
			layout = *layouts.begin();
		}
		++i;
	}
	if (!layout) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
	return {i, layout};
}

std::vector<TypingPlan> KeyboardLayout::build_typing_plan(const std::string& text_) {
	std::u32string text = conv.from_bytes(text_);

	std::vector<TypingPlan> result;

	for (size_t i = 0; i < text.size();) {
		// step 1: detemine the layout of the next chunk

		// special case: if the chunk starts with a common char: always use US layout,
		// we don't even care about the actual layout on the target system
		if (IsCommonChar(text[i])) {
			size_t j = FindNextNotCommonChar(text, i);
			result.push_back({&KeyboardLayout::US, "", "", "", conv.to_bytes(text.substr(i, j-i))});
			i = j;
			continue;
		}

		// general case: find the longest possible sequence, that can be
		// typed with a single layout
		auto [j, layout] = ChooseLayout(text, i);

		// step 2: prepare a TypingPlan structure

		size_t right_bound = FindRightBound(text, i, j);
		size_t right_core = std::min(i+3, right_bound);
		std::u32string core = text.substr(i, right_core-i);
		std::u32string tail = text.substr(right_core, j-right_core);

		size_t left_bound = FindLeftBound(text, i);
		size_t left_core = std::max(int64_t(left_bound), int64_t(i)-int64_t(3-core.size()));
		std::u32string prefix = text.substr(left_core, i-left_core);
		std::u32string postfix = layout->identifying_seq.substr(0, 3-core.size()-prefix.size());

		result.push_back({layout,
			conv.to_bytes(prefix),
			conv.to_bytes(core),
			conv.to_bytes(postfix),
			conv.to_bytes(tail)});

		i = j;
	}

	return result;
}

bool KeyboardLayout::can_be_typed_using_a_single_layout(const std::string& text_) {
	std::u32string text = conv.from_bytes(text_);
	size_t n = 0;
	for (size_t i = 0; i < text.size();) {
		auto [j, layout] = ChooseLayout(text, i);
		i = j;
		++n;
	}
	return n <= 1;
}

std::vector<KeyboardCommand> KeyboardLayout::type(const std::string& text_) const {
	std::u32string text = conv.from_bytes(text_);

	std::vector<KeyboardCommand> result;

	bool shift_holded = false;

	for (char32_t ch: text) {
		const CharDesc& char_desc = find_char_desc(ch);
		if (char_desc.hold_shift && !shift_holded) {
			result.push_back({KeyboardAction::Hold, KeyboardButton::LEFTSHIFT});
			shift_holded = true;
		}
		if (!char_desc.hold_shift && shift_holded) {
			result.push_back({KeyboardAction::Release, KeyboardButton::LEFTSHIFT});
			shift_holded = false;
		}
		result.push_back({KeyboardAction::Hold, char_desc.button});
		result.push_back({KeyboardAction::Release, char_desc.button});
	}

	if (shift_holded) {
		result.push_back({KeyboardAction::Release, KeyboardButton::LEFTSHIFT});
	}

	return result;
}

std::vector<KeyboardCommand> KeyboardLayout::clear(const std::string& text_) const {
	std::u32string text = conv.from_bytes(text_);

	std::vector<KeyboardCommand> result;
	for (size_t i = 0; i < text.size(); ++i) {
		result.push_back({KeyboardAction::Hold, KeyboardButton::BACKSPACE});
		result.push_back({KeyboardAction::Release, KeyboardButton::BACKSPACE});
	}
	return result;
}

const CharDesc& KeyboardLayout::find_char_desc(char32_t ch) const {
	auto it = char_map.find(ch);
	if (it != char_map.end()) {
		return it->second;
	}
	throw std::runtime_error("Unable to type the character " + conv.to_bytes(ch) + " with " + name + " layout");
}

bool KeyboardLayout::can_type(char32_t ch) const {
	return char_map.find(ch) != char_map.end();
}