
#pragma once
#include "Utils.hpp"

struct IsoId {
	IsoId() = delete;

	IsoId(const std::string& iso_spec, const fs::path& starting_point) {
		auto at_pos = iso_spec.find("@");
		if (at_pos != std::string::npos) {
			pool = iso_spec.substr(0, at_pos);
			name = iso_spec.substr(at_pos + 1);
		} else {
			name = iso_spec;
		}

		//if local - get canonical form at once
		if (!pool.length()) {
			fs::path iso_path(name);
			if (iso_path.is_relative()) {
				iso_path = starting_point / iso_path;
			}

			if (!fs::exists(iso_path)) {
				throw std::runtime_error("specified iso file doesn't exist: " + name);
			}

			iso_path = fs::canonical(iso_path);

			if (!fs::is_regular_file(iso_path)) {
				throw std::runtime_error(std::string("specified iso is not a regular file: ") + name);
			}

			name = iso_path.generic_string();
		}
	}
	
	std::string name;
	std::string pool;
};
