
#pragma once

#include <Windows.h>

namespace winapi {

struct Event {
	Event(BOOL manual_reset, BOOL is_signaled);
	~Event();

	Event(Event&& other);
	Event& operator=(Event&& other);

	void set();
	DWORD wait(DWORD ms);

private:
	HANDLE handle = NULL;
};

}
