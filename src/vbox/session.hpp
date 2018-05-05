
#pragma once

#include "api.hpp"

namespace vbox {

struct Session {
	Session(ISession* handle);
	~Session();

	Session(const Session&) = delete;
	Session& operator=(const Session&) = delete;
	Session(Session&& other);
	Session& operator=(Session&& other);

	std::string name() const;

	ISession* handle = nullptr;
};

}
