
#include "event.hpp"
#include "error.hpp"
#include <stdexcept>

namespace cl {

Event::Event(cl_event handle): _handle(handle) {
	try {
		if (!_handle) {
			throw std::runtime_error("nullptr");
		}
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Event::~Event() {
	if (_handle) {
		clReleaseEvent(_handle);
		_handle = nullptr;
	}
}

Event::Event(const Event& other): _handle(other._handle) {
	try {
		throw_if_failed(clRetainEvent(_handle));
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Event& Event::operator=(const Event& other) {
	try {
		throw_if_failed(clRetainEvent(other._handle));
		_handle = other._handle;
		return *this;
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

Event::Event(Event&& other): _handle(other._handle) {
	other._handle = nullptr;
}

Event& Event::operator=(Event&& other) {
	std::swap(_handle, other._handle);
	return *this;
}

void wait(const std::vector<Event>& events) {
	try {
		static_assert(sizeof(cl_event) == sizeof(Event));
		throw_if_failed(clWaitForEvents(events.size(), (cl_event*)events.data()));
	} catch (const std::exception&) {
		throw_with_nested(std::runtime_error(__FUNCSIG__));
	}
}

}
