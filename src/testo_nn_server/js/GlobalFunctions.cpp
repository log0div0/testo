
#include "GlobalFunctions.hpp"
#include "Tensor.hpp"
#include <iostream>

namespace js {

Value print(ContextRef ctx, const ValueRef this_val, const std::vector<ValueRef>& args) {
	auto& output = ctx.get_stdout();
	for (size_t i = 0; i < args.size(); i++) {
		if (i != 0) {
			std::cout << ' ';
		}
		output << args[i];
	}
	output << std::endl;
	return ctx.new_undefined();
}


Value find_text(ContextRef ctx, const ValueRef this_val, const std::vector<ValueRef>& args) {
	if (args.size() > 1) {
		throw std::runtime_error("Invalid arguments count in find_text");
	}

	nn::TextTensor tensor = nn::find_text(ctx.image());
	if (args.size() == 1) {
		std::string text = std::string(args.at(0));
		tensor = tensor.match_text(ctx.image(), text);
	}
	
	return TextTensor(ctx, tensor);
}

Value find_img(ContextRef ctx, const ValueRef this_val, const std::vector<ValueRef>& args) {
	if (args.size() != 1) {
		throw std::runtime_error("Invalid arguments count in find_img");
	}

	std::string img_path = args.at(0);

	stb::Image<stb::RGB> ref_image = ctx.env()->get_ref_image(img_path);
	
	nn::ImgTensor tensor = nn::find_img(ctx.image(), &ref_image);
	return ImgTensor(ctx, tensor);
}

}
