
#include "HyperVEnvironment.hpp"

fs::path HyperVEnvironment::testo_dir() const {
	return ".";
}

void HyperVEnvironment::setup() {
	Environment::setup();

	_putenv_s("HYPERV", "1");
}
