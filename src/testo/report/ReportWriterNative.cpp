
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

static std::string to_string(IR::Test::SnapshotPolicy policy) {
	switch (policy) {
		case IR::Test::SnapshotPolicy::Unknown:
			return "unknown";
		case IR::Test::SnapshotPolicy::Always:
			return "always";
		case IR::Test::SnapshotPolicy::Never:
			return "never";
		case IR::Test::SnapshotPolicy::Auto:
			return "auto";
		default:
			throw std::runtime_error("Invalid SnapshotPolicy value");
	}
}

static std::string now() {
	auto start_timestamp_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	std::stringstream ss;
	ss << std::put_time(std::localtime(&start_timestamp_t), "%FT%T%z");
	return ss.str();
}

static std::string to_string(std::chrono::system_clock::time_point tp) {
	auto t = std::chrono::system_clock::to_time_t(tp);
	std::stringstream ss;
	ss << std::put_time(std::localtime(&t), "%FT%T%z");
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
	current_launch_meta["num_of_tests_runs"] = tests_runs.size();

	for (auto test: tests) {
		if (test->is_up_to_date()) {
			current_launch_meta["up_to_date_tests"].push_back(test->name());
		}
	}
}

void ReportWriterNative::test_skip_begin(const std::shared_ptr<IR::TestRun>& test_run) {
	current_launch_meta["skipped_tests"].push_back(test_run->test->name());
}

void ReportWriterNative::test_skip_end(const std::shared_ptr<IR::TestRun>& test_run) {
}

void ReportWriterNative::test_begin(const std::shared_ptr<IR::TestRun>& test_run) {
	current_launch_meta["executed_tests"].push_back(test_run->test->name());
}

void ReportWriterNative::report_prefix(const std::shared_ptr<IR::TestRun>& test_run) {
	if (test_run) {
		report(test_run, fmt::format("[{}] ", test_run->test->name()));
	} else {
		report(test_run, fmt::format(">>> "));
	}
}

void ReportWriterNative::report_raw(const std::shared_ptr<IR::TestRun>& test_run, const std::string& text) {
	report(test_run, text);
}

void ReportWriterNative::test_end(const std::shared_ptr<IR::TestRun>& test_run) {
}

void ReportWriterNative::launch_end() {
	current_launch_meta["stop_timestamp"] = now();
}

nlohmann::json ReportWriterNative::to_json(const std::shared_ptr<IR::Test>& test) {
	return {
		{"name", test->name()},
		{"parents", test->parent_names()},
		{"snapshots_policy", to_string(test->snapshot_policy())},
		{"description", test->description()},
		{"cksum", test->cksum},
		{"cache_status", to_string(test->cache_status())},
		{"source_file", test->get_source_file_path()}
	};
}

nlohmann::json ReportWriterNative::to_json(const std::shared_ptr<IR::TestRun>& test_run) {
	return {
		{"id", test_run->id},
		{"test_name", test_run->test->name()},
		{"start_timestamp", to_string(test_run->start_timestamp)},
		{"stop_timestamp", to_string(test_run->stop_timestamp)},
		{"exec_status", to_string(test_run->exec_status)},
		{"failure_message", test_run->failure_message},
		{"failure_stacktrace", test_run->failure_stacktrace},
		{"failure_category", test_run->failure_category},
		{"found_bugs", test_run->found_bugs},
	};
}
