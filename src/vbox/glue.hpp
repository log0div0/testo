
#pragma once

#include <VBoxCAPIGlue/VBoxCAPIGlue.h>

namespace vbox {

struct Glue {
	Glue();
	~Glue();

	Glue(const Glue&) = delete;
	Glue& operator=(const Glue&) = delete;
};

}
