
#include "Host.hpp"

namespace vir {

Connect connect_open(const std::string& uri) {
	return virConnectOpen(uri.c_str());
}

}