
#pragma once

#include "api.hpp"
#include "additions_facility.hpp"
#include "guest_session.hpp"

#include <vector>

namespace vbox {

struct Guest {
	Guest(IGuest* handle);
	~Guest();

	Guest(const Guest& other) = delete;
	Guest& operator=(const Guest& other) = delete;

	Guest(Guest&& other);
	Guest& operator=(Guest&& other);

	GuestSession create_session(const std::string& user,
		const std::string& password,
		const std::string& domain = {},
		const std::string& name = {});
	
	std::vector<AdditionsFacility> facilities() const;

	IGuest* handle = nullptr;
};

}
