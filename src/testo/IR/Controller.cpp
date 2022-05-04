
#include "Controller.hpp"
#include "Action.hpp"
#include <fmt/format.h>

namespace IR {

std::string Controller::name() const {
	return Id(ast_node->name, stack).value();
}

bool Controller::is_defined() const {
	return fs::exists(main_file());
}

std::string Controller::note_was_declared_here() const {
	std::stringstream ss;
	ss << macro_call_stack << std::string(ast_node->begin())
		<< ": note: the " << type() << " " << name() << " was declared here";
	return ss.str();
}

bool Controller::has_snapshot(const std::string& snapshot, bool hypervisor_snapshot_needed) {
	fs::path metadata_file = get_metadata_dir();
	metadata_file /= id() + "_" + snapshot;
	bool has_metadata_snapshot = fs::exists(metadata_file);
	if (hypervisor_snapshot_needed) {
		return has_metadata_snapshot && has_hypervisor_snapshot(snapshot);
	} else {
		return has_metadata_snapshot;
	}
}

bool Controller::check_metadata_version() {
	try {
		fs::path metadata_file = get_metadata_dir();
		metadata_file /= id() + "_" + "_init";
		auto metadata = read_metadata_file(metadata_file);
		if (!metadata.count("opaque")) {
			return false;
		}

		if (!metadata.count("metadata_version")) {
			return false;
		}

		if (!metadata.at("metadata_version").is_number()) {
			return false;
		}

		return metadata.at("metadata_version").get<int>() == TESTO_CURRENT_METADATA_VERSION;
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("getting snapshot cksum error"));
	}
}

std::string Controller::get_snapshot_cksum(const std::string& snapshot) {
	try {
		fs::path metadata_file = get_metadata_dir();
		metadata_file /= id() + "_" + snapshot;
		auto metadata = read_metadata_file(metadata_file);
		if (!metadata.count("cksum")) {
			throw std::runtime_error("Can't find cksum field in snapshot metadata " + snapshot);
		}

		return metadata.at("cksum").get<std::string>();
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error("getting snapshot cksum error"));
	}
}

bool Controller::has_key(const std::string& key) {
	try {
		auto metadata = read_metadata_file(main_file());
		return metadata.count(key);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Checking metadata with key {}", key)));
	}
}

nlohmann::json Controller::get_metadata(const std::string& key) const {
	return get_metadata(main_file(), key);
}

void Controller::set_metadata(const std::string& key, const nlohmann::json& value) {
	set_metadata(main_file(), key, value);
}

nlohmann::json Controller::get_metadata(const fs::path& file, const std::string& key) {
	try {
		auto metadata = read_metadata_file(file);
		if (!metadata.count(key)) {
			throw std::runtime_error("Requested key is not present in metadata");
		}
		return metadata.at(key);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Getting metadata with key {}", key)));
	}
}

void Controller::set_metadata(const fs::path& file, const std::string& key, const nlohmann::json& value) {
	try {
		auto metadata = read_metadata_file(file);
		metadata[key] = value;
		write_metadata_file(file, metadata);
	} catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(fmt::format("Setting metadata with key {}", key)));
	}
}

nlohmann::json Controller::read_metadata_file(const fs::path& file) {
	std::ifstream metadata_file_stream(file.generic_string());
	if (!metadata_file_stream) {
		throw std::runtime_error("Can't read metadata file " + file.generic_string());
	}

	nlohmann::json result = nlohmann::json::parse(metadata_file_stream);
	metadata_file_stream.close();
	return result;
}


void Controller::write_metadata_file(const fs::path& file, const nlohmann::json& metadata) {
	std::ofstream metadata_file_stream(file.generic_string());
	if (!metadata_file_stream) {
		throw std::runtime_error("Can't write metadata file " + file.generic_string());
	}

	metadata_file_stream << metadata;
	metadata_file_stream.close();
}

fs::path Controller::main_file() const {
	fs::path result = get_metadata_dir();
	result = result / id();
	return result;
}

}
