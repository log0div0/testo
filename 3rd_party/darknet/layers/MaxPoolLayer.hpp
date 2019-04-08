
#pragma once

#include "../Layer.hpp"
#include <inipp.hh>

namespace darknet {

struct MaxPoolLayer: Layer {
	MaxPoolLayer(const inipp::inisection& section,
		size_t batch,
		size_t w,
		size_t h,
		size_t c);

	virtual void load_weights(FILE* fp) override {};
	virtual void save_weights(FILE* fp) const override {};
};

}
