
#pragma once

#include "api.hpp"
#include <string>

namespace vbox {

struct AdditionsFacility {
	AdditionsFacility(IAdditionsFacility* handle);
	~AdditionsFacility();

	AdditionsFacility(const AdditionsFacility& other) = delete;
	AdditionsFacility& operator=(const AdditionsFacility& other) = delete;

	AdditionsFacility(AdditionsFacility&& other);
	AdditionsFacility& operator=(AdditionsFacility&& other);

	std::string name() const;
	AdditionsFacilityType type() const;

	IAdditionsFacility* handle = nullptr;
};

}
