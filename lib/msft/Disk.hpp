
#pragma once

#include "Partition.hpp"

namespace msft {

struct Disk {
	Disk(wmi::WbemClassObject disk_, wmi::WbemServices services_);

	std::string friendlyName() const;
	void initialize(uint16_t partitionStyle = 1);
	void clear();

	void createPartition();
	std::vector<Partition> partitions() const;

	wmi::WbemClassObject disk;
	wmi::WbemServices services;
};

}
