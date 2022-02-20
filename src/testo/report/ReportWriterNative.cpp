
#include "ReportWriterNative.hpp"
#include <fmt/format.h>
#include <iomanip>

static std::string to_string(IR::TestRun::ExecStatus status) {
	switch (status) {
		case IR::TestRun::ExecStatus::Unknown:
			return "unknown";
		case IR::TestRun::ExecStatus::Passed:
			return "passed";
		case IR::TestRun::ExecStatus::Failed:
			return "failed";
		case IR::TestRun::ExecStatus::Skipped:
			return "skipped";
		default:
			throw std::runtime_error("Invalid ExecStatus value");
	}
}

static std::string to_string(IR::Test::CacheStatus status) {
	switch (status) {
		case IR::Test::CacheStatus::Unknown:
			return "unknown";
		case IR::Test::CacheStatus::Empty:
			return "empty";
		case IR::Test::CacheStatus::OK:
			return "ok";
		case IR::Test::CacheStatus::Miss:
			return "miss";
		default:
			throw std::runtime_error("Invalid CacheStatus value");
	}
}

static std::string now() {
	auto start_timestamp_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	std::stringstream ss;
	ss << std::put_time(std::localtime(&start_timestamp_t), "%FT%T%z");
	return ss.str();
}

ReportWriterNative::ReportWriterNative(const ReportConfig& config): ReportWriter(config) {
	current_launch_meta["id"] = generate_uuid_v4();
	current_launch_meta["testo_version"] = TESTO_VERSION;
	current_launch_meta["config"] = nlohmann::json::object();
	config.dump(current_launch_meta["config"]);
}

void ReportWriterNative::launch_begin(const std::vector<std::shared_ptr<IR::Test>>& tests,
	const std::vector<std::shared_ptr<IR::TestRun>>& tests_runs)
{
	current_launch_meta["start_timestamp"] = now();
	current_launch_meta["working_dir"] = fs::current_path();
	current_launch_meta["executed_tests"] = nlohmann::json::array();
	current_launch_meta["up_to_date_tests"] = nlohmann::json::array();
	current_launch_meta["skipped_tests"] = nlohmann::json::array();

	for (auto test: tests) {
		if (test->is_up_to_date()) {
			current_launch_meta["up_to_date_tests"].push_back(test->name());
		}
	}
}

void ReportWriterNative::test_skip(const std::shared_ptr<IR::TestRun>& test_run) {
	current_test_run_meta = nlohmann::json::object();
	current_test_run_meta["id"] = generate_uuid_v4();
	current_test_run_meta["test_name"] = current_test_run->test->name();
	current_test_run_meta["exec_status"] = to_string(current_test_run->exec_status);

	current_launch_meta["skipped_tests"].push_back(test_run->test->name());
}

void ReportWriterNative::test_begin(const std::shared_ptr<IR::TestRun>& test_run) {
	current_test_run = test_run;

	current_test_run_meta = nlohmann::json::object();
	current_test_run_meta["id"] = generate_uuid_v4();
	current_test_run_meta["test_name"] = current_test_run->test->name();
	current_test_run_meta["start_timestamp"] = now();

	current_launch_meta["executed_tests"].push_back(current_test_run_meta["id"]);
}

void ReportWriterNative::report_prefix() {
	if (current_test_run) {
		report(fmt::format("[{}] ", current_test_run->test->name()));
	} else {
		report(fmt::format(">>> "));
	}
}

void ReportWriterNative::report_raw(const std::string& text) {
	report(text);
}

void ReportWriterNative::test_end() {
	current_test_run_meta["stop_timestamp"] = now();
	current_test_run_meta["exec_status"] = to_string(current_test_run->exec_status);
	current_test_run_meta["failure_message"] = current_test_run->failure_message;
	current_test_run_meta["failure_stacktrace"] = current_test_run->failure_stacktrace;
	current_test_run_meta["failure_category"] = current_test_run->failure_category;

	current_test_run = nullptr;
}

void ReportWriterNative::launch_end() {
	current_launch_meta["stop_timestamp"] = now();
}

nlohmann::json ReportWriterNative::to_json(const std::shared_ptr<IR::Test>& test) {
	return {
		{"name", test->name()},
		{"parents", test->parent_names()},
		{"snapshots_needed", test->snapshots_needed()},
		{"description", test->description()},
		{"cksum", test->cksum},
		{"cache_status", to_string(test->cache_status())},
		{"source_file", test->get_source_file_path()}
	};
}
