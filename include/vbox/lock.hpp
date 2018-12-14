
#pragma once

#include "machine.hpp"
#include "session.hpp"

namespace vbox {

struct Lock {
	Lock(Machine& machine, Session& session, LockType lock_type);
	~Lock();

	Lock() = delete;

	Lock(const Lock&) = delete;
	Lock& operator=(const Lock&) = delete;
	Lock(Lock&& other);
	Lock& operator=(Lock&& other);

	IMachine* machine;
	ISession* session;
	SessionState prev_state;
};

}
