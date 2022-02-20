
#pragma once

#include <nlohmann/json.hpp>

struct EnvironmentConfig {
	std::string nn_server_endpoint = "127.0.0.1:8156";

	void validate() const;
	virtual void dump(nlohmann::json& j) const;
};

struct VisitorSemanticConfig {
	std::string prefix;

	void validate() const;

	virtual void dump(nlohmann::json& j) const;
};

struct ReportConfig {
	std::string report_folder;

	void validate() const;

	virtual void dump(nlohmann::json& j) const;
};

enum class ReportFormat {
	NativeLocal,
	NativeRemote,
	Allure
};

struct ReporterConfig: ReportConfig {
	bool html = false;
	std::string report_format = "native_local";
	ReportFormat get_report_format() const;

	void validate() const;

	virtual void dump(nlohmann::json& j) const;
};

struct VisitorInterpreterConfig: ReporterConfig {
	bool stop_on_fail = false;
	bool assume_yes = false;
	std::string invalidate;
	bool dry = false;

	void validate() const;

	virtual void dump(nlohmann::json& j) const;
};

struct TestNameFilter {
	enum class Type {
		test_spec,
		exclude
	};
	Type type;
	std::string pattern;

	bool validate_test_name(const std::string& name) const;
};

void to_json(nlohmann::json& j, const TestNameFilter& filter);

struct ProgramConfig: VisitorSemanticConfig, VisitorInterpreterConfig, EnvironmentConfig {
	std::string target;

	std::vector<TestNameFilter> test_name_filters;

	std::vector<std::string> params_names;
	std::vector<std::string> params_values;

	bool validate_test_name(const std::string& name) const;
	void validate() const;

	virtual void dump(nlohmann::json& j) const;
};
