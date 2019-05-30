
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
		auto deadline = std::chrono::system_clock::now() + 10s;
		do {
			SessionState_T result = SessionState_Null;
			throw_if_failed(IMachine_get_SessionState(machine, &result));
			if ((result == SessionState_Unlocked) ||
				((result == SessionState_Locked) && (lock_type == LockType_Shared)))
			{
				do {
					try {
						throw_if_failed(IMachine_LockMachine(machine, session, lock_type));
						return;
					} catch (const std::exception& error) {
						std::this_thread::sleep_for(100ms);
					}
				} while (std::chrono::system_clock::now() < deadline);

				throw std::runtime_error("Failed to lock machine");
			}
			std::this_thread::sleep_for(100ms);
		} while (std::chrono::system_clock::now() < deadline);

		throw std::runtime_error("Failed to lock machine");
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Lock::~Lock() {
	try {
		throw_if_failed(ISession_UnlockMachine(session));
	}
	catch (const std::exception& error) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
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
