
#pragma once

#include <CL/cl.h>
#include <vector>

namespace cl {

struct Event {
	Event(cl_event handle);
	~Event();

	Event(const Event& other);
	Event& operator=(const Event& other);
	Event(Event&& other);
	Event& operator=(Event&& other);

	cl_event handle() const { return _handle; }

private:
	cl_event _handle = nullptr;
};

void wait(const std::vector<Event>& events);

}
