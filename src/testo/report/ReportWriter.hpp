
#pragma once

#include <nlohmann/json.hpp>
#include "../IR/Test.hpp"
#include "../Configs.hpp"
#include <fstream>

struct ReportWriter {
	ReportWriter(const ReportConfig& config) {}
	virtual ~ReportWriter() {}

	virtual void launch_begin(const std::vector<std::shared_ptr<IR::Test>>& tests,
		const std::vector<std::shared_ptr<IR::TestRun>>& tests_runs) {}

	virtual void test_skip(const std::shared_ptr<IR::TestRun>& test_run) {}
	virtual void test_begin(const std::shared_ptr<IR::TestRun>& test_run) {}
	virtual void report_prefix() {}
	virtual void report(const std::string& text) {}
	virtual void report_raw(const std::string& text) {}
	virtual void report_screenshot(const stb::Image<stb::RGB>& screenshot) {}
	virtual void test_end() {}

	virtual void launch_end() {}
};
