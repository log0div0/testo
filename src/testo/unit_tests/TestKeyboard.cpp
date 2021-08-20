
#include <catch.hpp>
#include "../Keyboard.hpp"

using KA = KeyboardAction;
using KB = KeyboardButton;

TEST_CASE("keyboard type hello") {
	std::vector<KeyboardCommand> actual = KeyboardManager().type("hello");
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
	std::vector<KeyboardCommand> actual = KeyboardManager().type("Hello Мир");
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
		{KA::Hold, KB::SPACE},
		{KA::Release, KB::SPACE},
		{KA::Hold, KB::LEFTSHIFT},
		{KA::Hold, KB::V},
		{KA::Release, KB::V},
		{KA::Release, KB::LEFTSHIFT},
		{KA::Hold, KB::B},
		{KA::Release, KB::B},
		{KA::Hold, KB::H},
		{KA::Release, KB::H},
	};
	REQUIRE(actual == expected);
}
