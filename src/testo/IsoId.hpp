
#pragma once

struct IsoId {
	IsoId() = delete;

	IsoId(const std::string& iso_spec) {
		auto at_pos = iso_spec.find("@");
		if (at_pos != std::string::npos) {
			pool = iso_spec.substr(0, at_pos);
			name = iso_spec.substr(at_pos + 1);
		} else {
			name = iso_spec;
		}
	}

	std::string name;
	std::string pool;
};
