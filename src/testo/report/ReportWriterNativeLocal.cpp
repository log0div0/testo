#include "ReportWriterNativeLocal.hpp"
#include <fmt/format.h>
#include <iomanip>

ReportWriterNativeLocal::ReportWriterNativeLocal(const ReportConfig& config): ReportWriterNative(config) {
	report_folder = config.report_folder;
}

const std::string tag_file = ".testo_report_folder";

void ReportWriterNativeLocal::launch_begin(const std::vector<std::shared_ptr<IR::Test>>& tests,
	const std::vector<std::shared_ptr<IR::TestRun>>& tests_runs)
{
	ReportWriterNative::launch_begin(tests, tests_runs);

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
	fs::create_directories(report_folder / "launches" / current_launch_meta.at("id").get<std::string>());
	std::ofstream(report_folder / tag_file);
	current_launch_output_file = std::ofstream(report_folder / "launches" / current_launch_meta.at("id").get<std::string>() / "log.txt", std::ios_base::app);

	for (auto& test: tests) {
		initialize_test(test);
	}
}

void ReportWriterNativeLocal::test_begin(const std::shared_ptr<IR::TestRun>& test_run) {
	ReportWriterNative::test_begin(test_run);
	fs::create_directories(report_folder / "tests_runs" / test_run->id);
	current_test_run_output_file = std::ofstream(report_folder / "tests_runs" / test_run->id / "log.txt");
}

void ReportWriterNativeLocal::report(const std::shared_ptr<IR::TestRun>& test_run, const std::string& text) {
	if (test_run) {
		current_test_run_output_file << text;
	} else {
		current_launch_output_file << text;
	}
}

void ReportWriterNativeLocal::report_screenshot(const std::shared_ptr<IR::TestRun>& test_run, const stb::Image<stb::RGB>& screenshot) {
	screenshot.write_png((report_folder / "tests_runs" / test_run->id / "screenshot.png").generic_string());
}

void ReportWriterNativeLocal::test_end(const std::shared_ptr<IR::TestRun>& test_run) {
	ReportWriterNative::test_end(test_run);
	current_test_run_output_file.close();
	std::ofstream file(report_folder / "tests_runs" / test_run->id / "meta.json");
	file << to_json(test_run).dump(2);
}

void ReportWriterNativeLocal::launch_end() {
	ReportWriterNative::launch_end();
	auto path = fs::absolute(report_folder / "launches" / current_launch_meta.at("id").get<std::string>() / "meta.json");
	std::ofstream file(path);
	file << current_launch_meta.dump(2);
}

void ReportWriterNativeLocal::initialize_test(const std::shared_ptr<IR::Test>& test) {
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
		file << to_json(test).dump(2);
	}
}
