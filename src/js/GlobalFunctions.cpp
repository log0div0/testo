
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

	nlohmann::json request = {
		{"type", "ref_image_request"},
		{"data", img_path}
	};

	ctx.channel()->send_response(request);
	std::unique_ptr<Message> ref_image;
	try {
		ref_image = ctx.channel()->receive_message();
	} catch (const std::exception& error) {
		throw std::runtime_error("Couldn't get the ref image: " + std::string(error.what()));
	}

	if (ref_image->header["screenshot"].get<ImageSize>().total_size() == 0) {
		throw std::runtime_error("RefImage is empty");
	}

	if (auto p = dynamic_cast<RefImage*>(ref_image.get())) {
		nn::ImgTensor tensor = nn::find_img(ctx.image(), &p->screenshot);
		return ImgTensor(ctx, tensor);
	} else {
		throw std::runtime_error("Got the wrong type of message instead of RefImage");
	}
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
