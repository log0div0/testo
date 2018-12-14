
#include <vbox/array_out.hpp>

namespace vbox {

ArrayOut::~ArrayOut() {
	if (data) {
		api->pfnArrayOutFree(data);
	}
}

ArrayOutIface::~ArrayOutIface() {
	if (ifaces) {
		api->pfnArrayOutFree(ifaces);
	}
}

}
