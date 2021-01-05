
#include "HyperVEnvironment.hpp"

fs::path HyperVEnvironment::testo_dir() const {
	return ".";
}

void HyperVEnvironment::setup() {
	_putenv_s("HYPERV", "1");
}

void HyperVEnvironment::cleanup() {

}
