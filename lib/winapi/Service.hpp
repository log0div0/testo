
#pragma once

#include <Windows.h>

namespace winapi {

struct Service {
	Service(SC_HANDLE handle_);
	~Service();

	Service(Service&& other);
	Service& operator=(Service&& other);

	void start();
	SERVICE_STATUS control(DWORD signal);
	SERVICE_STATUS queryStatus() const;

private:
	SC_HANDLE handle = NULL;
};

}
