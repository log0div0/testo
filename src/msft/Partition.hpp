
#pragma once

#include "Volume.hpp"

namespace msft {

struct Partition {
	Partition(wmi::WbemClassObject partition_, wmi::WbemServices services_);

	void deleteObject();
	std::vector<std::string> getAccessPaths() const;
	std::vector<std::string> accessPaths() const;
	Volume volume() const;
	void addAccessPath(const std::string& path);

	wmi::WbemClassObject partition;
	wmi::WbemServices services;
};

}
