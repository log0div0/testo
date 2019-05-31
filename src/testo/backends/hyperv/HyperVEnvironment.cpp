
#include "HyperVEnvironment.hpp"

void HyperVEnvironment::setup() {
	_putenv_s("HYPERV", "1");
}

void HyperVEnvironment::cleanup() {

}
