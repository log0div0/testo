
#pragma once

#include "session.hpp"

namespace vbox {

struct Unlocker {
	Unlocker(Session& session): session(session) {}
	~Unlocker() {
		try {
			session.unlock_machine();
		} catch (...) {
			// деструктор не может бросать исключение
		}
	}

	Unlocker(const Unlocker&) = delete;
	Unlocker& operator=(const Unlocker&) = delete;
	Unlocker(Unlocker&& other) = delete;
	Unlocker& operator=(Unlocker&& other) = delete;

	Session& session;
};

}
