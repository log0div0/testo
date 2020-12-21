
#include "GlobalFunctions.hpp"
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
	if (args.size() > 1) {
		throw std::runtime_error("Invalid arguments count in find_text");
	}

	nn::TextTensor tensor = nn::find_text(ctx.image());
	if (args.size() == 1) {
		std::string text = args.at(0);
		tensor = tensor.match_text(ctx.image(), text);
	}
	return TextTensor(ctx, tensor);
}

Value find_img(ContextRef ctx, const ValueRef this_val, const std::vector<ValueRef>& args) {
	if (args.size() != 1) {
		throw std::runtime_error("Invalid arguments count in find_img");
	}

	std::string img_path = args.at(0);

	nn::ImgTensor tensor = nn::find_img(ctx.image(), img_path);
	return ImgTensor(ctx, tensor);
}

Value find_homme3(ContextRef ctx, const ValueRef this_val, const std::vector<ValueRef>& args) {
	if (args.size() != 1) {
		throw std::runtime_error("Invalid arguments count in find_homm3");
	}

	std::string id = args.at(0);

	nn::Homm3Tensor tensor = nn::find_homm3(ctx.image(), id);
	return Homm3Tensor(ctx, tensor);
}

}
