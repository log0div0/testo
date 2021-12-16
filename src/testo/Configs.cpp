
#include "Configs.hpp"
#include <wildcards.hpp>
#include <ghc/filesystem.hpp>
#include <set>

namespace fs = ghc::filesystem;

std::string EnvironmentConfig::nn_server_ip() const {
	auto semicolon_pos = nn_server_endpoint.find(":");
	return nn_server_endpoint.substr(0, semicolon_pos);
}

std::string EnvironmentConfig::nn_server_port() const {
	auto semicolon_pos = nn_server_endpoint.find(":");
	return nn_server_endpoint.substr(semicolon_pos + 1, nn_server_endpoint.length() - 1);
}

void EnvironmentConfig::validate() const {
	auto semicolon_pos = nn_server_endpoint.find(":");
	if (semicolon_pos == std::string::npos) {
		throw std::runtime_error("ip_port string is malformed: no semicolon");
	}

	auto port = nn_server_port();

	try {
		auto uport = std::stoul(port);
		if (uport > 65535) {
			throw std::runtime_error("");
		}
	} catch (const std::exception& error) {
		throw std::runtime_error(std::string("nn_server port doesn't seem to be valid: ") + port);
	}
}

void EnvironmentConfig::dump(nlohmann::json& j) const {
	j["nn_server_ip"] = nn_server_ip();
	j["nn_server_port"] = nn_server_port();
}

void VisitorSemanticConfig::validate() const {

}

void VisitorSemanticConfig::dump(nlohmann::json& j) const {
	j["prefix"] = prefix;
}

void ReportConfig::validate() const {
}

void ReportConfig::dump(nlohmann::json& j) const {
	j["report_folder"] = report_folder;
}

ReportFormat ReporterConfig::get_report_format() const {
	if (report_format == "native") {
		return ReportFormat::Native;
	} else if (report_format == "allure") {
		return ReportFormat::Allure;
	} else {
		throw std::runtime_error("Invalid report format: " + report_format);
	}
}

void ReporterConfig::validate() const {
	ReportConfig::validate();
	get_report_format(); // will throw on error
}

void ReporterConfig::dump(nlohmann::json& j) const {
	ReportConfig::dump(j);
	j["report_format"] = report_format;
	j["html"] = html;
}

void VisitorInterpreterConfig::validate() const {
	ReporterConfig::validate();
}

void VisitorInterpreterConfig::dump(nlohmann::json& j) const {
	ReporterConfig::dump(j);
	j["stop_on_fail"] = stop_on_fail;
	j["assume_yes"] = assume_yes;
	j["invalidate"] = invalidate;
	j["dry"] = dry;
}

bool TestNameFilter::validate_test_name(const std::string& name) const {
	switch (type) {
		case Type::test_spec:
			return wildcards::match(name, pattern);
		case Type::exclude:
			return !wildcards::match(name, pattern);
		default:
			throw std::runtime_error("Should not be there");
	}
}

void to_json(nlohmann::json& j, const TestNameFilter& filter) {
	switch (filter.type) {
		case TestNameFilter::Type::test_spec:
			j["type"] = "test_spec";
			break;
		case TestNameFilter::Type::exclude:
			j["type"] = "exclude";
			break;
		default:
			throw std::runtime_error("Should not be there");
	}
	j["pattern"] = filter.pattern;
}

bool ProgramConfig::validate_test_name(const std::string& name) const {
	for (auto& filter: test_name_filters) {
		if (!filter.validate_test_name(name)) {
			return false;
		}
	}
	return true;
}

void ProgramConfig::validate() const {
	if (!fs::exists(target)) {
		throw std::runtime_error("Error: target doesn't exist: " + target);
	}

	std::set<std::string> unique_param_names;

	for (size_t i = 0; i < params_names.size(); ++i) {
		auto result = unique_param_names.insert(params_names[i]);
		if (!result.second) {
			throw std::runtime_error("Error: param \"" + params_names[i] + "\" is defined multiple times as a command line argument");
		}
	}

	VisitorSemanticConfig::validate();
	VisitorInterpreterConfig::validate();
	EnvironmentConfig::validate();
}

void ProgramConfig::dump(nlohmann::json& j) const {
	VisitorSemanticConfig::dump(j);
	VisitorInterpreterConfig::dump(j);
	EnvironmentConfig::dump(j);

	j["target"] = target;
	j["test_name_filters"] = test_name_filters;
	auto params = nlohmann::json::object();
	for (size_t i = 0; i < params_names.size(); ++i) {
		params[params_names.at(i)] = params_values.at(i);
	}
	j["params"] = params;
	j["use_cpu"] = use_cpu;
}