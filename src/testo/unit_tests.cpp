
#define CATCH_CONFIG_RUNNER
#include <catch.hpp>
#include "Parser.hpp"

int main( int argc, char* argv[] ) {
#ifdef WIN32
	SetConsoleOutputCP(CP_UTF8);
#endif
	int result = Catch::Session().run( argc, argv );
	return result;
}

TEST_CASE("Parse wait 1") {
	Parser parser(".", "{ wait \"hello world\" timeout 65ms interval 32s; }");
	auto block = parser.action_block();
	auto wait = std::dynamic_pointer_cast<AST::Wait>(block->actions.at(0));
	REQUIRE(wait->timeout->text() == "65ms");
	REQUIRE(wait->interval->text() == "32s");
}

TEST_CASE("Parse wait 2") {
	Parser parser(".", "{ wait \"hello world\" interval 32s timeout 65ms; }");
	auto block = parser.action_block();
	auto wait = std::dynamic_pointer_cast<AST::Wait>(block->actions.at(0));
	REQUIRE(wait->timeout->text() == "65ms");
	REQUIRE(wait->interval->text() == "32s");
}

