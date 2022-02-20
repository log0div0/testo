
#pragma once

#include "ReportWriter.hpp"

struct ReportWriterNative: ReportWriter {
	ReportWriterNative(const ReportConfig& config);

	virtual void launch_begin() override;

	virtual void initialize_test_run(const std::shared_ptr<IR::TestRun>& test_run) override;
	virtual void initialize_up_to_date_test(const std::shared_ptr<IR::Test>& test) override;

	virtual void test_skip(const std::shared_ptr<IR::TestRun>& test_run) override;
	virtual void test_begin(const std::shared_ptr<IR::TestRun>& test_run) override;
	virtual void report_prefix() override;
	virtual void report(const std::string& text) override;
	virtual void report_raw(const std::string& text) override;
	virtual void report_screenshot(const stb::Image<stb::RGB>& screenshot) override;
	virtual void test_end() override;

	virtual void launch_end() override;

private:
	void initialize_test(const std::shared_ptr<IR::Test>& test);

	fs::path report_folder;

	std::string current_launch_id;
	std::ofstream current_launch_output_file;
	nlohmann::json current_launch_meta;

	std::shared_ptr<IR::TestRun> current_test_run;
	std::string current_test_run_id;
	std::ofstream current_test_run_output_file;
	nlohmann::json current_test_run_meta;
};
