
#include "api.hpp"
#include <stdexcept>
#include <set>

#if defined(__linux__) || defined(__FreeBSD__)
# define DYNLIB_NAME "VBoxXPCOMC.so"
#elif defined(WIN32)
# define DYNLIB_NAME "VBoxCAPI.dll"
#elif defined(__APPLE__)
# define DYNLIB_NAME "VBoxXPCOMC.dylib"
#else
#error "Port me"
#endif

void setenv(const std::string& name, const std::string& value) {
#ifdef WIN32
	_putenv_s(name.c_str(), value.c_str());
#else
	setenv(name.c_str(), value.c_str(), 1);
#endif
}

namespace vbox {

PCVBOXCAPI api = nullptr;

API::API() {
	std::set<std::string> dirs;
	if (getenv("VBOX_APP_HOME")) {
		dirs.insert(getenv("VBOX_APP_HOME"));
	} else {
#if defined(__linux__)
		dirs.insert("/usr/lib/virtualbox");
#elif defined(__FreeBSD__)
		dirs.insert("/usr/local/lib/virtualbox");
#elif defined(WIN32)
		if (getenv("ProgramFiles")) {
			dirs.insert(getenv("ProgramFiles") + std::string("\\Oracle\\VirtualBox"));
		}
		dirs.insert("C:\\Program Files\\Oracle\\VirtualBox");
#elif defined(__APPLE__)
		dirs.insert("/Applications/VirtualBox.app/Contents/MacOS");
#else
#error "port me"
#endif
	}

	for (auto& dir: dirs) {
		if (!getenv("VBOX_APP_HOME")) {
			setenv("VBOX_APP_HOME", dir.c_str());
		}

		std::string path = getenv("PATH");

		try {
			setenv("PATH", path + ";" + dir);

			_dynlib = DynLib(dir + "/" + DYNLIB_NAME);

			PFNVBOXGETCAPIFUNCTIONS get_api = (PFNVBOXGETCAPIFUNCTIONS)_dynlib.sym(VBOX_GET_CAPI_FUNCTIONS_SYMBOL_NAME);

			api = get_api(VBOX_CAPI_VERSION);
			if (!api) {
				throw std::runtime_error("get_api doesn't work");
			}

			if ((VBOX_CAPI_MAJOR(api->uVersion) != VBOX_CAPI_MAJOR(VBOX_CAPI_VERSION)) ||
				(VBOX_CAPI_MINOR(api->uVersion) < VBOX_CAPI_MINOR(VBOX_CAPI_VERSION))) {
				throw std::runtime_error("Incompatible vbox version");
			}

			return;
		} catch (const std::exception&) {
			setenv("PATH", path);
			continue;
		}
	}

	throw std::runtime_error("Failed to initialize virtual box api");
}

API::~API() {
	api = nullptr;
}

}
