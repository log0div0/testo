
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

TEST_CASE("keyboard split Hello") {
	std::vector<TextChunk> actual = KeyboardLayout::split_text_by_layout("Hello");
	std::vector<TextChunk> expected = {
		{&KeyboardLayout::US, "Hello"}
	};
	REQUIRE(actual == expected);
}

TEST_CASE("keyboard split He№llo Мир!") {
	std::vector<TextChunk> actual = KeyboardLayout::split_text_by_layout("He№llo Мир!");
	std::vector<TextChunk> expected = {
		{&KeyboardLayout::US, "He"},
		{&KeyboardLayout::RU, "№"},
		{&KeyboardLayout::US, "llo "},
		{&KeyboardLayout::RU, "Мир!"}
	};
	REQUIRE(actual == expected);
}
