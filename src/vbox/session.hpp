
#pragma once

#include "api.hpp"
#include "machine.hpp"

namespace vbox {

struct Session {
	Session(ISession* handle);
	~Session();

	Session(const Session&) = delete;
	Session& operator=(const Session&) = delete;
	Session(Session&& other);
	Session& operator=(Session&& other);

	std::string name() const;

	Machine machine() const;

	void unlock_machine();

	ISession* handle = nullptr;
};

}
