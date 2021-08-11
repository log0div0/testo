
#include "../nn_service/Messages.hpp"
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

	ctx.channel()->send(create_ref_image_request(img_path));
	nlohmann::json response;
	try {
		response = ctx.channel()->recv();
	} catch (const std::exception& error) {
		throw std::runtime_error("Couldn't get the ref image: " + std::string(error.what()));
	}
	
	check_for_error(response);

	auto type = response.at("type").get<std::string>();
	if (type != "ref_image") {
		throw std::runtime_error("Unexpected message type instead of \"ref_image\": " + type);
	}

	stb::Image<stb::RGB> ref_image;
	ref_image = get_image(response);
	
	nn::ImgTensor tensor = nn::find_img(ctx.image(), &ref_image);
	return ImgTensor(ctx, tensor);
}

Value find_homme3(ContextRef ctx, const ValueRef this_val, const std::vector<ValueRef>& args) {
	if (args.size() > 1) {
		throw std::runtime_error("Invalid arguments count in find_homm3");
	}

	nn::Homm3Tensor tensor = nn::find_homm3(ctx.image());
	if (args.size() == 1) {
		std::string class_id = args.at(0);
		tensor = tensor.match_class(ctx.image(), class_id);
	}
	return Homm3Tensor(ctx, tensor);
}

}
