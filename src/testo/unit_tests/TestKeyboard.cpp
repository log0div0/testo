
#include <catch.hpp>
#include "../Keyboard.hpp"

using KA = KeyboardAction;
using KB = KeyboardButton;

TEST_CASE("keyboard type hello") {
	std::vector<KeyboardCommand> actual = KeyboardLayout::US.type("hello");
	std::vector<KeyboardCommand> expected = {
		{KA::Hold, KB::H},
		{KA::Release, KB::H},
		{KA::Hold, KB::E},
		{KA::Release, KB::E},
		{KA::Hold, KB::L},
		{KA::Release, KB::L},
		{KA::Hold, KB::L},
		{KA::Release, KB::L},
		{KA::Hold, KB::O},
		{KA::Release, KB::O},
	};
	REQUIRE(actual == expected);
}

TEST_CASE("keyboard type Hello") {
	std::vector<KeyboardCommand> actual = KeyboardLayout::US.type("Hello");
	std::vector<KeyboardCommand> expected = {
		{KA::Hold, KB::LEFTSHIFT},
		{KA::Hold, KB::H},
		{KA::Release, KB::H},
		{KA::Release, KB::LEFTSHIFT},
		{KA::Hold, KB::E},
		{KA::Release, KB::E},
		{KA::Hold, KB::L},
		{KA::Release, KB::L},
		{KA::Hold, KB::L},
		{KA::Release, KB::L},
		{KA::Hold, KB::O},
		{KA::Release, KB::O},
	};
	REQUIRE(actual == expected);
}

TEST_CASE("keyboard split - simple") {
	std::vector<TypingPlan> actual = KeyboardLayout::build_typing_plan("Hello");
	std::vector<TypingPlan> expected = {
		{&KeyboardLayout::US, "", 		"Hel",		"",			"lo"},
	};
	REQUIRE(actual == expected);
}

TEST_CASE("keyboard split - two layouts") {
	std::vector<TypingPlan> actual = KeyboardLayout::build_typing_plan("He№llo Мир!");
	std::vector<TypingPlan> expected = {
		{&KeyboardLayout::US, "",		"He",		"f",		""},
		{&KeyboardLayout::RU, "He",		"№",		"",			""},
		{&KeyboardLayout::US, "",		"llo",		"",			" "},
		{&KeyboardLayout::RU, "",		"Мир",		"",			"!"},
	};
	REQUIRE(actual == expected);
}

TEST_CASE("keyboard split - multiline") {
	std::vector<TypingPlan> actual = KeyboardLayout::build_typing_plan("Hello\nФJ\nМир!");
	std::vector<TypingPlan> expected = {
		{&KeyboardLayout::US, "",		"Hel",		"",			"lo\n"},
		{&KeyboardLayout::RU, "",		"Ф",		"яф",		""},
		{&KeyboardLayout::US, "Ф",		"J",		"f",		"\n"},
		{&KeyboardLayout::RU, "",		"Мир",		"",			"!"},
	};
	REQUIRE(actual == expected);
}

TEST_CASE("keyboard split - numbers") {
	std::vector<TypingPlan> actual = KeyboardLayout::build_typing_plan("12345 Hello");
	std::vector<TypingPlan> expected = {
		{&KeyboardLayout::US, "", 		"",			"",			"12345 "},
		{&KeyboardLayout::US, "", 		"Hel",		"",			"lo"},
	};
	REQUIRE(actual == expected);
}

TEST_CASE("keyboard split - numbers 2") {
	std::vector<TypingPlan> actual = KeyboardLayout::build_typing_plan("Hello 12345");
	std::vector<TypingPlan> expected = {
		{&KeyboardLayout::US, "", 		"Hel",		"",			"lo 12345"},
	};
	REQUIRE(actual == expected);
}

TEST_CASE("keyboard split - spaces") {
	std::vector<TypingPlan> actual = KeyboardLayout::build_typing_plan("H   ello");
	std::vector<TypingPlan> expected = {
		{&KeyboardLayout::US, "", 		"H",		"fw",		"   ello"},
	};
	REQUIRE(actual == expected);
}

TEST_CASE("keyboard split - spaces 2") {
	std::vector<TypingPlan> actual = KeyboardLayout::build_typing_plan("Hello   М");
	std::vector<TypingPlan> expected = {
		{&KeyboardLayout::US, "", 		"Hel",		"",			"lo   "},
		{&KeyboardLayout::RU, "", 		"М",		"яф",		""},
	};
	REQUIRE(actual == expected);
}
