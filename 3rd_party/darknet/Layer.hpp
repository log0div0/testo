
#pragma once

#include "include/darknet.h"

namespace darknet {

struct Layer: layer {
	Layer(): layer() {

	}

	virtual ~Layer() {};

	Layer(const Layer& other) = delete;
	Layer& operator=(const Layer& other) = delete;

	virtual void load_weights(FILE* fp) = 0;
	virtual void save_weights(FILE* fp) const = 0;
};

}
