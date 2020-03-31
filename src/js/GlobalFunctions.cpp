
#include "GlobalFunctions.hpp"
#include "nn/OCR.hpp"
#include <iostream>

namespace js {

Value js_print(ContextRef ctx, const ValueRef this_val, const std::vector<ValueRef>& args) {
	for (size_t i = 0; i < args.size(); i++) {
		if (i != 0) {
			std::cout << ' ';
		}
		std::cout << args[i];
	}
	std::cout << std::endl;
	return ctx.new_undefined();
}


Value detect_text(ContextRef ctx, const ValueRef this_val, const std::vector<ValueRef>& args) {
	if (args.size() > 3) {
		throw std::runtime_error("Invalid arguments count in detect_text");
	}

	std::string text, color, background_color;
	text = std::string(args.at(0));

	if (args.size() > 1) {
		color = std::string(args.at(1));
	}

	if (args.size() > 2) {
		background_color = std::string(args.at(2));
	}

	auto result = nn::OCR(ctx.image()).search(text, color, background_color);
	auto array = ctx.new_array(result.size());

	for (size_t i = 0; i < result.size(); ++i) {
		auto& rect = result[i];
		auto obj = ctx.new_object_class(nn_rect_class_id);
		if (obj.is_exception()) {
			throw std::runtime_error("Can't create nn::Rect class object");
		}
		// TODO obj.set_opaque(std::make_shared<nn::Rect>(result[i]));
		obj.set_property_str("x", ctx.new_int32(rect.center_x()));
		obj.set_property_str("y", ctx.new_int32(rect.center_y()));

		array.set_property_uint32(i, obj);
	}

	return array;
}

}
