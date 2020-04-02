
#pragma once

#include "Context.hpp"
#include <nn/OCR.hpp>

namespace js {

struct Tensor: Value {
	static void register_class(ContextRef ctx);

	Tensor(ContextRef ctx, const nn::Tensor& tensor);
};

}
