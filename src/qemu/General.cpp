
#include "General.hpp"

namespace qemu {

Connect ConnectOpen(const std::string& uri) {
	return virConnectOpen(uri.c_str());
}

}