
#include "lock.hpp"
#include "throw_if_failed.hpp"

#include <chrono>
#include <thread>

#include <iostream>

using namespace std::chrono_literals;

namespace vbox {

Lock::Lock(Machine& machine_, Session& session_, LockType lock_type): machine(machine_.handle),
	session(session_.handle)
{
	try {
		SessionState_T result = SessionState_Null;
		throw_if_failed(IMachine_get_SessionState(machine, &result));

		while ((result == SessionState_Unlocking) || (result == SessionState_Spawning)) {
			std::this_thread::sleep_for(40ms);
			throw_if_failed(IMachine_get_SessionState(machine, &result));
		}

		prev_state = result;
		throw_if_failed(IMachine_LockMachine(machine, session, lock_type));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Lock::~Lock() {
	try {
		throw_if_failed(ISession_UnlockMachine(session));
		if (prev_state == SessionState_Unlocked) {

			SessionState_T result = SessionState_Null;
			throw_if_failed(IMachine_get_SessionState(machine, &result));

			while (result != SessionState_Unlocked) {
				std::this_thread::sleep_for(40ms);
				throw_if_failed(IMachine_get_SessionState(machine, &result));
			}
		}
	}
	catch (const std::exception& error) {
		std::cout << error.what() << std::endl;
	}
}

Lock::Lock(Lock&& other): machine(other.machine), session(other.session) {
	other.machine = nullptr;
	other.session = nullptr;
}

Lock& Lock::operator=(Lock&& other) {
	std::swap(machine, other.machine);
	std::swap(session, other.session);
	return *this;
}

}
