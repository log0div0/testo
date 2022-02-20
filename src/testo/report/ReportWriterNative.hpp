
#pragma once

#include "ReportWriter.hpp"

struct ReportWriterNative: ReportWriter {
	ReportWriterNative(const ReportConfig& config);

	virtual void launch_begin(const std::vector<std::shared_ptr<IR::Test>>& tests,
		const std::vector<std::shared_ptr<IR::TestRun>>& tests_runs) override;

	virtual void test_skip(const std::shared_ptr<IR::TestRun>& test_run) override;
	virtual void test_begin(const std::shared_ptr<IR::TestRun>& test_run) override;
	virtual void report_prefix() override;
	virtual void report_raw(const std::string& text) override;
	virtual void test_end() override;

	virtual void launch_end() override;

protected:
	static nlohmann::json to_json(const std::shared_ptr<IR::Test>& test);

	nlohmann::json current_launch_meta;
	nlohmann::json current_test_run_meta;

	std::shared_ptr<IR::TestRun> current_test_run;
};
