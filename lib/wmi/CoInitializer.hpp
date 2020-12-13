
#pragma once

namespace wmi {

struct CoInitializer {
	CoInitializer();
	~CoInitializer();

	void initalize_security();
};

}
