
#pragma once

#include "session.hpp"

namespace vbox {

struct Unlocker {
	Unlocker(Session& session): session(session) {}
	~Unlocker() {
		session.unlock_machine();
	}

	Unlocker(const Unlocker&) = delete;
	Unlocker& operator=(const Unlocker&) = delete;
	Unlocker(Unlocker&& other) = delete;
	Unlocker& operator=(Unlocker&& other) = delete;

	Session& session;
};

}
