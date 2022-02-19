
#pragma once

#include "ReportWriterNative.hpp"

struct ReportWriterNativeLocal: ReportWriterNative {
	ReportWriterNativeLocal(const ReportConfig& config);

	virtual void launch_begin(const std::vector<std::shared_ptr<IR::Test>>& tests,
		const std::vector<std::shared_ptr<IR::TestRun>>& tests_runs) override;

	virtual void test_begin(const std::shared_ptr<IR::TestRun>& test_run) override;
	virtual void report(const std::shared_ptr<IR::TestRun>& test_run, const std::string& text) override;
	virtual void report_screenshot(const std::shared_ptr<IR::TestRun>& test_run, const stb::Image<stb::RGB>& screenshot) override;
	virtual void test_end(const std::shared_ptr<IR::TestRun>& test_run) override;

	virtual void launch_end() override;

private:
	void initialize_test(const std::shared_ptr<IR::Test>& test);

	fs::path report_folder;

	std::ofstream current_launch_output_file;
	std::ofstream current_test_run_output_file;
};
