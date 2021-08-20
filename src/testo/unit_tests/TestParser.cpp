
#include <catch.hpp>
#include "../Parser.hpp"

void TestParseStringifyActions(const std::string& str) {
	auto block = Parser(".", str).action_block();
	auto str2 = block->to_string();
	REQUIRE(str == str2);
}

TEST_CASE("parse action wait") {
	TestParseStringifyActions("{ wait \"hello world\" interval 32s timeout 65ms; }");
	TestParseStringifyActions("{ wait \"hello world\" timeout 65ms interval 32s; }");
	TestParseStringifyActions("{ wait \"hello world\"; }");
	TestParseStringifyActions("{ wait \"hello world\" timeout 65ms; }");
	TestParseStringifyActions("{ wait \"hello world\" interval 32s; }");
	TestParseStringifyActions("{ wait \"hello world\" timeout \"${SOME_PARAM}\" interval \"some_prefix_${SOME_OTHER_PARAM}\"; }");
}

TEST_CASE("parse action macro call") {
	TestParseStringifyActions("{ some_macro(); }");
	TestParseStringifyActions("{ some_macro(\"10\", \"hello world\"); }");
}

TEST_CASE("parse action mouse click") {
	TestParseStringifyActions("{ mouse click \"Next\".from_right(0).center_bottom(); }");
}
