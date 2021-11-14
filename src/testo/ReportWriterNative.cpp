
#include "ReportWriterNative.hpp"
#include <fmt/format.h>
#include <iomanip>

ReportWriterNative::ReportWriterNative(const ReportConfig& config): ReportWriter(config) {
	report_folder = config.report_folder;

	current_launch_meta["config"] = nlohmann::json::object();
	config.dump(current_launch_meta["config"]);
}

const std::string tag_file = ".testo_report_folder";

static std::string now() {
	auto start_timestamp_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	std::stringstream ss;
	ss << std::put_time(std::localtime(&start_timestamp_t), "%FT%T%z");
	return ss.str();
}

void ReportWriterNative::launch_begin() {
	current_launch_meta["start_timestamp"] = now();
	current_launch_meta["working_dir"] = fs::current_path();
	current_launch_meta["tests_runs"] = nlohmann::json::array();
	current_launch_meta["up_to_date_tests"] = nlohmann::json::array();

	if (fs::exists(report_folder)) {
		if (!fs::is_directory(report_folder)) {
			throw std::runtime_error("Specified report folder " + report_folder.generic_string() + " is not a folder");
		}
		if (!fs::is_empty(report_folder)) {
			if (!fs::exists(report_folder / tag_file)) {
				throw std::runtime_error("Specified report folder " + report_folder.generic_string() + " is not a report folder");
			}
		}
	}
	current_launch_id = generate_uuid_v4();
	fs::create_directories(report_folder / "launches" / current_launch_id);
	std::ofstream(report_folder / tag_file);
	current_launch_output_file = std::ofstream(report_folder / "launches" / current_launch_id / "log.txt", std::ios_base::app);
}

void ReportWriterNative::initialize_test_run(const std::shared_ptr<IR::TestRun>& test_run) {
	initialize_test(test_run->test);
}

void ReportWriterNative::initialize_up_to_date_test(const std::shared_ptr<IR::Test>& test) {
	current_launch_meta["up_to_date_tests"].push_back(test->name());
	initialize_test(test);
}

void ReportWriterNative::test_skip(const std::shared_ptr<IR::TestRun>& test_run) {
	// do nothing
}

void ReportWriterNative::test_begin(const std::shared_ptr<IR::TestRun>& test_run) {
	current_test_run = test_run;
	current_test_run_id = generate_uuid_v4();
	current_launch_meta["tests_runs"].push_back(current_test_run_id);
	fs::create_directories(report_folder / "tests_runs" / current_test_run_id);
	current_test_run_output_file = std::ofstream(report_folder / "tests_runs" / current_test_run_id / "log.txt");
}

void ReportWriterNative::report_prefix() {
	if (current_test_run) {
		report(fmt::format("[{}] ", current_test_run->test->name()));
	} else {
		report(fmt::format("[???] "));
	}
}

void ReportWriterNative::report(const std::string& text) {
	if (current_test_run) {
		current_test_run_output_file << text;
	} else {
		current_launch_output_file << text;
	}
}

void ReportWriterNative::report_raw(const std::string& text) {
	report(text);
}

void ReportWriterNative::report_screenshot(const stb::Image<stb::RGB>& screenshot) {
	screenshot.write_png((report_folder / "tests_runs" / current_test_run_id / "screenshot.png").generic_string());
}

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

static std::string to_string(std::chrono::system_clock::time_point tp) {
	auto t = std::chrono::system_clock::to_time_t(tp);
	std::stringstream ss;
	ss << std::put_time(std::localtime(&t), "%FT%T%z");
	return ss.str();
}

void ReportWriterNative::test_end() {
	current_test_run_output_file.close();
	std::ofstream file(report_folder / "tests_runs" / current_test_run_id / "meta.json");
	current_test_run_meta = {
		{"test_name", current_test_run->test->name()},
		{"exec_status", to_string(current_test_run->exec_status)},
		{"start_timestamp", to_string(current_test_run->start_timestamp)},
		{"stop_timestamp", to_string(current_test_run->stop_timestamp)}
	};
	file << current_test_run_meta.dump(2);
	current_test_run = nullptr;
}

void ReportWriterNative::launch_end() {
	current_launch_meta["stop_timestamp"] = now();
	auto path = fs::absolute(report_folder / "launches" / current_launch_id / "meta.json");
	std::ofstream file(path);
	file << current_launch_meta.dump(2);
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

void ReportWriterNative::initialize_test(const std::shared_ptr<IR::Test>& test) {
	fs::path path = report_folder / "tests" / test->name() / "meta.json";
	if (fs::exists(path)) {
		std::ifstream file(path);
		if (!file.is_open()) {
			throw std::runtime_error("Failed to open file " + path.generic_string());
		}
		nlohmann::json j;
		file >> j;
		if (j.at("cksum") != test->cksum) {
			throw std::runtime_error("Can't write the report into an existing report folder because \"" + test->name() + "\" test's checksum has been changed");
		}
	} else {
		fs::create_directories(report_folder / "tests" / test->name());
		std::ofstream file(path);
		nlohmann::json meta = {
			{"name", test->name()},
			{"parents", test->parent_names()},
			{"snapshots_needed", test->snapshots_needed()},
			{"description", test->description()},
			{"cksum", test->cksum},
			{"cache_status", to_string(test->cache_status())}
		};
		file << meta.dump(2);
	}
}
