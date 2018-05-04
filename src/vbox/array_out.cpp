
#include "array_out.hpp"

namespace vbox {

ArrayOut::~ArrayOut() {
	if (values) {
		api->pfnArrayOutFree(values);
	}
}

}
