
#include "array_out.hpp"

namespace vbox {

ArrayOut::~ArrayOut() {
	if (data) {
		api->pfnArrayOutFree(data);
	}
}

ArrayOut::ArrayOut(ArrayOut&& other): data(other.data), data_size(other.data_size) {
	other.data = nullptr;
	other.data_size = 0;
}

ArrayOut& ArrayOut::operator=(ArrayOut&& other) {
	std::swap(data, other.data);
	std::swap(data_size, other.data_size);
	return *this;
}

ArrayOutIface::~ArrayOutIface() {
	if (ifaces) {
		api->pfnArrayOutFree(ifaces);
	}
}

ArrayOutIface::ArrayOutIface(ArrayOutIface&& other): ifaces(other.ifaces), ifaces_count(other.ifaces_count) {
	other.ifaces = nullptr;
	other.ifaces_count = 0;
}

ArrayOutIface& ArrayOutIface::operator=(ArrayOutIface&& other) {
	std::swap(ifaces, other.ifaces);
	std::swap(ifaces_count, other.ifaces_count);
	return *this;
}

}
