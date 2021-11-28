
#include "ReportWriterAllure.hpp"
#include "IR/Program.hpp"
#include <chrono>

std::string escape(const std::string& input) {
	std::string output;
	output.reserve(input.size());
	for (char ch: input) {
		switch (ch) {
			case '"':
				output += "&quot;";
				break;
			case '\'':
				output += "&apos;";
				break;
			case '<':
				output += "&lt;";
				break;
			case '>':
				output += "&gt;";
				break;
			case '&':
				output += "&amp;";
				break;
			default:
				output.push_back(ch);
				break;
		}
	}
	return output;
}

Attachment::Attachment(const fs::path& report_folder, const std::string& str) {
	title = "Output";
	type = "text/plain";
	source = generate_uuid_v4() + "-attachment.txt";
	fs::path path = report_folder / source;
	std::ofstream file(path);
	file << str;
}

Attachment::Attachment(const fs::path& report_folder, const stb::Image<stb::RGB>& screenshot) {
	title = "Screenshot";
	type = "image/png";
	source = generate_uuid_v4() + "-attachment.png";
	fs::path path = report_folder / source;
	screenshot.write_png(path.generic_string());
}

std::string Attachment::to_xml() const {
	std::string xml = fmt::format(R"(<attachment title="{}" source="{}" type="{}"/>)", title, source, type);
	return xml;
}

std::string to_string(std::chrono::system_clock::time_point tp) {
	return std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()).count());
}

std::string Step::to_xml() const {
	std::string xml = fmt::format(R"(<step start="{}" stop="{}" status="{}">
		<name>{}</name>
		<title>{}</title>)", to_string(start), to_string(stop), status, escape(name), escape(title));

	xml += R"(<attachments>)";

	for (auto& attachment: attachments) {
		xml += attachment.to_xml();
	}

	xml += R"(</attachments>)";

	xml += R"(</step>)";

	return xml;
}

std::string Failure::to_xml() const {
	std::string xml = fmt::format(R"(<failure>
		<message>{}</message>
		<stack-trace>{}</stack-trace>
	</failure>)", escape(message), escape(stacktrace));
	return xml;
}

std::string Label::to_xml() const {
	std::string xml = fmt::format(R"(<label name="{}" value="{}"/>)", name, value);
	return xml;
}

std::string TestSuite::to_xml() const {
	std::string xml = fmt::format(R"(<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<ns2:test-suite xmlns:ns2="urn:model.allure.qatools.yandex.ru">
	<name>{}</name>
	<test-cases>)", escape(name));

	for (auto& testcase: testcases) {
		xml += testcase.to_xml();
	}

	xml += R"(
	</test-cases>
	<labels/>
</ns2:test-suite>
)";

	return xml;
}

TestCase::TestCase(const std::shared_ptr<IR::Test>& test) {
	name = test->name();
	title = test->title();
	description = test->description();

	add_label(test, "feature");
	add_label(test, "story");
	add_label(test, "severity");
}

void TestCase::add_label(const std::shared_ptr<IR::Test>& test, const std::string& name) {
	if (test->attrs.count(name)) {
		labels.push_back({name, test->attrs.at(name)});
	}
}

std::string TestCase::to_xml() const {
	std::string xml = fmt::format(R"(
		<test-case status="{}" start="{}" stop="{}">
			<name>{}</name>)", status, to_string(start), to_string(stop), escape(name));

	if (title.size()) {
		xml += fmt::format(R"(<title>{}</title>)", escape(title));
	}
	if (description.size()) {
		xml += fmt::format(R"(<description type="text">{}</description>)", escape(description));
	}

	xml += failure.to_xml();

	xml += R"(<steps>)";

	for (auto& step: steps) {
		xml += step.to_xml();
	}

	xml += R"(
			</steps>)";

	xml += R"(<labels>)";

	for (auto& label: labels) {
		xml += label.to_xml();
	}

	xml += R"(
			</labels>
		</test-case>)";

	return xml;
}

ReportWriterAllure::ReportWriterAllure(const ReportConfig& config): ReportWriter(config) {
	report_folder = config.report_folder;
}

void ReportWriterAllure::launch_begin() {
	fs::create_directories(report_folder);
	write_environment_file();
	write_categories_file();
}

void ReportWriterAllure::initialize_up_to_date_test(const std::shared_ptr<IR::Test>& test) {
	TestCase testcase(test);
	testcase.status = "unknown";
	testcase.failure.message = "This test is cached";
	testcase.failure.stacktrace = test->cksum_input.str();
	testcase.start = std::chrono::system_clock::now();
	testcase.stop = std::chrono::system_clock::now();

	TestSuite& testsuite = testsuites[test->get_source_file_path().parent_path()];
	testsuite.testcases.push_back(testcase);
}

void ReportWriterAllure::test_skip(const std::shared_ptr<IR::TestRun>& test_run) {
	TestCase testcase(test_run->test);
	testcase.status = "skipped";
	testcase.failure.message = "Some of the parent tests has failed";
	for (auto& parent: test_run->get_unsuccessful_parents_names()) {
		if (testcase.failure.stacktrace.size()) {
			testcase.failure.stacktrace += ", ";
		}
		testcase.failure.stacktrace += parent;
	}
	testcase.start = std::chrono::system_clock::now();
	testcase.stop = std::chrono::system_clock::now();

	TestSuite& testsuite = testsuites[test_run->test->get_source_file_path().parent_path()];
	testsuite.testcases.push_back(testcase);
}

void ReportWriterAllure::test_begin(const std::shared_ptr<IR::TestRun>& test_run) {
	current_test_run = test_run;
	current_testcase = TestCase(test_run->test);
	current_testcase.start = std::chrono::system_clock::now();
}

void ReportWriterAllure::report_prefix() {
	if (current_testcase.steps.size()) {
		current_testcase.steps.back().status = "passed";
		current_testcase.steps.back().stop = std::chrono::system_clock::now();
		if (current_testcase.steps.back().raw.size()) {
			current_testcase.steps.back().attachments.push_back(Attachment(report_folder, current_testcase.steps.back().raw));
		}
	}
	Step step;
	step.start = std::chrono::system_clock::now();
	current_testcase.steps.push_back(step);
}

void ReportWriterAllure::report(const std::string& text) {
	if (current_testcase.steps.size()) {
		current_testcase.steps.back().title += text;
	}
}

void ReportWriterAllure::report_raw(const std::string& text) {
	if (current_testcase.steps.size()) {
		current_testcase.steps.back().raw += text;
	}
}

void ReportWriterAllure::report_screenshot(const stb::Image<stb::RGB>& screenshot) {
	if (current_testcase.steps.size()) {
		current_testcase.steps.back().attachments.push_back(Attachment(report_folder, screenshot));
	}
}

void ReportWriterAllure::test_end() {
	current_testcase.stop = std::chrono::system_clock::now();
	switch (current_test_run->exec_status) {
		case IR::TestRun::ExecStatus::Passed:
			current_testcase.status = "passed";
			switch (current_test_run->test->cache_status()) {
				case IR::Test::CacheStatus::Empty:
					current_testcase.failure.message = "No cache found";
					break;
				case IR::Test::CacheStatus::Miss:
					current_testcase.failure.message = "Cache miss";
					break;
				default:
					current_testcase.failure.message = "Something weird has happened with the cache";
					break;
			}
			current_testcase.failure.stacktrace = current_test_run->test->cksum_input.str();
			if (current_testcase.steps.size()) {
				current_testcase.steps.back().status = "passed";
			}
			break;
		case IR::TestRun::ExecStatus::Failed:
			current_testcase.status = "failed";
			current_testcase.failure.message = current_test_run->failure_message;
			current_testcase.failure.stacktrace = current_test_run->failure_stacktrace;
			if (current_testcase.steps.size()) {
				current_testcase.steps.back().status = "failed";
			}
			break;
		default:
			current_testcase.status = "broken";
			break;
	}
	if (current_testcase.steps.size()) {
		current_testcase.steps.back().stop = std::chrono::system_clock::now();
		if (current_testcase.steps.back().raw.size()) {
			current_testcase.steps.back().attachments.push_back(Attachment(report_folder, current_testcase.steps.back().raw));
		}
	}
	TestSuite& testsuite = testsuites[current_test_run->test->get_source_file_path().parent_path()];
	testsuite.testcases.push_back(current_testcase);
}

fs::path build_most_common_dir(fs::path a, fs::path b) {
	fs::path result;
	auto A = a.begin();
	auto B = b.begin();
	for (; A != a.end() && B != b.end(); ++A, ++B) {
		if (*A != *B) {
			break;
		}
		result = result / *A;
	}
	return result;
}

std::string build_testuite_name(const fs::path& path) {
	std::string result;
	for (auto& part: path) {
		if (result.size()) {
			result += ".";
		}
		result += part;
	}
	return result;
}

void ReportWriterAllure::launch_end() {
	if (testsuites.size() == 0) {
		return;
	}
	fs::path top_dir = testsuites.begin()->first;
	for (auto& kv: testsuites) {
		top_dir = build_most_common_dir(top_dir, kv.first);
	}
	top_dir = top_dir.parent_path();
	for (auto& kv: testsuites) {
		TestSuite& testsuite = kv.second;
		testsuite.name = build_testuite_name(fs::relative(kv.first, top_dir));

		std::ofstream file(report_folder / (generate_uuid_v4() + "-testsuite.xml"));
		file << testsuite.to_xml();
	}
}

void ReportWriterAllure::write_environment_file() {
	std::ofstream file(report_folder / "environment.properties");
	for (auto& kv: IR::program->stack->params) {
		file << kv.first << "=" << kv.second << std::endl;
	}
}

void ReportWriterAllure::write_categories_file() {
	nlohmann::json j = {
		{
			{"name", "Passed tests"},
			{"matchedStatuses", { "passed" }},
		},
		{
			{"name", "Failed tests"},
			{"matchedStatuses", { "failed" }},
		},
		{
			{"name", "Skipped tests"},
			{"matchedStatuses", { "skipped" }},
		},
		{
			{"name", "Up-to-date tests"},
			{"matchedStatuses", { "unknown" }},
		},
	};

	std::ofstream file(report_folder / "categories.json");
	file << j.dump(2);
}
