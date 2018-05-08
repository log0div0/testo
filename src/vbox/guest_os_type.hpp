
#pragma once

#include "api.hpp"

namespace vbox {

struct GuestOSType {
	GuestOSType(IGuestOSType* handle);
	~GuestOSType();

	GuestOSType(const GuestOSType&) = delete;
	GuestOSType& operator=(const GuestOSType&) = delete;
	GuestOSType(GuestOSType&& other);
	GuestOSType& operator=(GuestOSType&& other);

	std::string id() const;
	std::string description() const;
	std::string family_id() const;
	std::string family_description() const;

	IGuestOSType* handle = nullptr;
};

std::ostream& operator<<(std::ostream& stream, const GuestOSType& guest_os_type);

}
