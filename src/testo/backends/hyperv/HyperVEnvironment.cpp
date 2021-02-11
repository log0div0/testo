
#include "HyperVEnvironment.hpp"
#include <winapi/RegKey.hpp>

fs::path HyperVEnvironment::testo_dir() const {
	winapi::RegKey regkey(HKEY_LOCAL_MACHINE, "SOFTWARE\\Testo Lang\\Testo", KEY_QUERY_VALUE);
	return fs::path(regkey.get_str("InstallDir")) / "metadata";
}

void HyperVEnvironment::setup() {
	Environment::setup();
}
