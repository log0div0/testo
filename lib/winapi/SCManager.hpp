
#pragma once

#include <Windows.h>
#include <string>
#include "Service.hpp"

namespace winapi {

struct SCManager {
	SCManager();
	~SCManager();

	SCManager(SCManager&& other);
	SCManager& operator=(SCManager&& other);

	Service service(const std::string& name, DWORD dwDesiredAccess);

private:
	SC_HANDLE handle = NULL;
};

}
