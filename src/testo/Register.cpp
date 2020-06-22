
#include "Register.hpp"

Register::~Register() {
	for (auto fdc: fdcs) {
		if (fdc.second->fd->is_mounted()) {
			fdc.second->fd->umount();
		}
	}
}
