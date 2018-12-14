
#pragma once

#include "api.hpp"

namespace vbox {

struct EventSource {
	EventSource() = default;
	EventSource(IEventSource* handle);
	~EventSource();

	EventSource(const EventSource&) = delete;
	EventSource& operator=(const EventSource&) = delete;
	EventSource(EventSource&& other);
	EventSource& operator=(EventSource&& other);

	IEventSource* handle = nullptr;
};

}
