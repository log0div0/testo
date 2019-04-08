
#pragma once

#include "include/darknet.h"

namespace darknet {

struct Layer: layer {
	virtual ~Layer() {
		free_layer(*this);
	};

	virtual void load_weights(FILE* fp) = 0;
	virtual void save_weights(FILE* fp) const = 0;
};

}
