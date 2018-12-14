
#include <vbox/event_source.hpp>
#include <stdexcept>
#include <vbox/throw_if_failed.hpp>

namespace vbox {

EventSource::EventSource(IEventSource* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

EventSource::~EventSource() {
	if (handle) {
		IEventSource_Release(handle);
	}
}

EventSource::EventSource(EventSource&& other): handle(other.handle) {
	other.handle = nullptr;
}

EventSource& EventSource::operator=(EventSource&& other) {
	std::swap(handle, other.handle);
	return *this;
}

}
