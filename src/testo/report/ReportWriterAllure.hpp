
#pragma once

#include "ReportWriter.hpp"

struct Attachment {
	Attachment(const fs::path& report_folder, const std::string& str);
	Attachment(const fs::path& report_folder, const stb::Image<stb::RGB>& screenshot);

	std::string to_xml() const;

	std::string title;
	std::string source;
	std::string type;
};

struct Step {
	std::string to_xml() const;
	void finish(const std::string& status_);

	std::chrono::system_clock::time_point start;
	std::chrono::system_clock::time_point stop;
	std::string name;
	std::string title;
	std::string status;

	std::vector<Attachment> attachments;

	std::string raw;
};

struct Failure {
	std::string to_xml() const;

	std::string message;
	std::string stacktrace;
};

struct Label {
	std::string to_xml() const;

	std::string name;
	std::string value;
};

struct TestCase {
	TestCase() = default;
	TestCase(const std::shared_ptr<IR::Test>& test);
	std::string to_xml() const;

	std::string name;
	std::string title;
	std::string description;
	std::string status;
	std::chrono::system_clock::time_point start;
	std::chrono::system_clock::time_point stop;
	std::vector<Label> labels;
	Failure failure;
	std::vector<Step> steps;

	void add_label(const std::shared_ptr<IR::Test>& test, const std::string& name);
};

struct TestSuite {
	std::string to_xml() const;

	std::string name;
	std::vector<TestCase> testcases;
};

struct ReportWriterAllure: ReportWriter {
	ReportWriterAllure(const ReportConfig& config);

	virtual void launch_begin() override;

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
	void write_environment_file();
	void write_categories_file();

	TestCase current_testcase;
	std::shared_ptr<IR::TestRun> current_test_run;

	std::map<fs::path, TestSuite> testsuites;

	fs::path report_folder;
};
