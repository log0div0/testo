
#include "GlobalFunctions.hpp"
#include "nn/OCR.hpp"
#include "Tensor.hpp"
#include <iostream>

namespace js {

Value print(ContextRef ctx, const ValueRef this_val, const std::vector<ValueRef>& args) {
	for (size_t i = 0; i < args.size(); i++) {
		if (i != 0) {
			std::cout << ' ';
		}
		std::cout << args[i];
	}
	std::cout << std::endl;
	return ctx.new_undefined();
}


Value find_text(ContextRef ctx, const ValueRef this_val, const std::vector<ValueRef>& args) {
	if (args.size() > 0) {
		throw std::runtime_error("Invalid arguments count in find_text");
	}

	nn::Tensor tensor = nn::find_text(ctx.image());
	return Tensor(ctx, tensor);
}

}
