
#pragma once

#include "machine.hpp"
#include "session.hpp"

namespace vbox {

template <LockType lock_type>
struct Lock {
	Lock(Machine& machine, Session& session): session(session) {
		machine.lock_machine(session, lock_type);
	}
	~Lock() {
		session.unlock_machine();
	}

	Lock(const Lock&) = delete;
	Lock& operator=(const Lock&) = delete;
	Lock(Lock&& other) = delete;
	Lock& operator=(Lock&& other) = delete;

	Session& session;
};

using WriteLock = Lock<LockType_Write>;
using SharedLock = Lock<LockType_Shared>;
using VMLock = Lock<LockType_VM>;

}
