
#include "Tensor.hpp"
#include "Point.hpp"
#include "FunctionsAdapters.hpp"
#include <iostream>

namespace js {

template <typename JSTensor>
Value size(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	typename JSTensor::Opaque* tensor = (typename JSTensor::Opaque*)this_val.get_opaque(JSTensor::class_id);
	return ctx.new_int32(tensor->size());
}

template <typename JSTensor>
Value x(ContextRef ctx, ValueRef this_val) {
	typename JSTensor::Opaque* tensor = (typename JSTensor::Opaque*)this_val.get_opaque(JSTensor::class_id);
	return ctx.new_int32(tensor->x());
}

template <typename JSTensor>
Value y(ContextRef ctx, ValueRef this_val) {
	typename JSTensor::Opaque* tensor = (typename JSTensor::Opaque*)this_val.get_opaque(JSTensor::class_id);
	return ctx.new_int32(tensor->y());
}

template <typename JSTensor>
Value left_top(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	typename JSTensor::Opaque* tensor = (typename JSTensor::Opaque*)this_val.get_opaque(JSTensor::class_id);
	if (args.size() != 0) {
		throw std::runtime_error("Invalid arguments count in Tensor::left_top");
	}
	return Point(ctx, tensor->left_top());
}

template <typename JSTensor>
Value left_bottom(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	typename JSTensor::Opaque* tensor = (typename JSTensor::Opaque*)this_val.get_opaque(JSTensor::class_id);
	if (args.size() != 0) {
		throw std::runtime_error("Invalid arguments count in Tensor::left_bottom");
	}
	return Point(ctx, tensor->left_bottom());
}

template <typename JSTensor>
Value right_top(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	typename JSTensor::Opaque* tensor = (typename JSTensor::Opaque*)this_val.get_opaque(JSTensor::class_id);
	if (args.size() != 0) {
		throw std::runtime_error("Invalid arguments count in Tensor::right_top");
	}
	return Point(ctx, tensor->right_top());
}

template <typename JSTensor>
Value right_bottom(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	typename JSTensor::Opaque* tensor = (typename JSTensor::Opaque*)this_val.get_opaque(JSTensor::class_id);
	if (args.size() != 0) {
		throw std::runtime_error("Invalid arguments count in Tensor::right_bottom");
	}
	return Point(ctx, tensor->right_bottom());
}

template <typename JSTensor>
Value center(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	typename JSTensor::Opaque* tensor = (typename JSTensor::Opaque*)this_val.get_opaque(JSTensor::class_id);
	if (args.size() != 0) {
		throw std::runtime_error("Invalid arguments count in Tensor::center");
	}
	return Point(ctx, tensor->center());
}

template <typename JSTensor>
Value center_top(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	typename JSTensor::Opaque* tensor = (typename JSTensor::Opaque*)this_val.get_opaque(JSTensor::class_id);
	if (args.size() != 0) {
		throw std::runtime_error("Invalid arguments count in Tensor::center_top");
	}
	return Point(ctx, tensor->center_top());
}

template <typename JSTensor>
Value center_bottom(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	typename JSTensor::Opaque* tensor = (typename JSTensor::Opaque*)this_val.get_opaque(JSTensor::class_id);
	if (args.size() != 0) {
		throw std::runtime_error("Invalid arguments count in Tensor::center_bottom");
	}
	return Point(ctx, tensor->center_bottom());
}

template <typename JSTensor>
Value left_center(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	typename JSTensor::Opaque* tensor = (typename JSTensor::Opaque*)this_val.get_opaque(JSTensor::class_id);
	if (args.size() != 0) {
		throw std::runtime_error("Invalid arguments count in Tensor::left_center");
	}
	return Point(ctx, tensor->left_center());
}

template <typename JSTensor>
Value right_center(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	typename JSTensor::Opaque* tensor = (typename JSTensor::Opaque*)this_val.get_opaque(JSTensor::class_id);
	if (args.size() != 0) {
		throw std::runtime_error("Invalid arguments count in Tensor::right_center");
	}
	return Point(ctx, tensor->right_center());
}

template <typename JSTensor>
Value from_top(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	typename JSTensor::Opaque* tensor = (typename JSTensor::Opaque*)this_val.get_opaque(JSTensor::class_id);
	if (args.size() != 1) {
		throw std::runtime_error("Invalid arguments count in Tensor::from_top");
	}
	int32_t index = args.at(0);
	return JSTensor(ctx, tensor->from_top(index));
}

template <typename JSTensor>
Value from_bottom(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	typename JSTensor::Opaque* tensor = (typename JSTensor::Opaque*)this_val.get_opaque(JSTensor::class_id);
	if (args.size() != 1) {
		throw std::runtime_error("Invalid arguments count in Tensor::from_bottom");
	}
	int32_t index = args.at(0);
	return JSTensor(ctx, tensor->from_bottom(index));
}

template <typename JSTensor>
Value from_left(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	typename JSTensor::Opaque* tensor = (typename JSTensor::Opaque*)this_val.get_opaque(JSTensor::class_id);
	if (args.size() != 1) {
		throw std::runtime_error("Invalid arguments count in Tensor::from_left");
	}
	int32_t index = args.at(0);
	return JSTensor(ctx, tensor->from_left(index));
}

template <typename JSTensor>
Value from_right(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	typename JSTensor::Opaque* tensor = (typename JSTensor::Opaque*)this_val.get_opaque(JSTensor::class_id);
	if (args.size() != 1) {
		throw std::runtime_error("Invalid arguments count in Tensor::from_right");
	}
	int32_t index = args.at(0);
	return JSTensor(ctx, tensor->from_right(index));
}

template <typename JSTensor>
Value match(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	typename JSTensor::Opaque* tensor = (typename JSTensor::Opaque*)this_val.get_opaque(JSTensor::class_id);
	if (args.size() != 1) {
		throw std::runtime_error("Invalid arguments count in Tensor::match");
	}
	std::string text = args.at(0);
	return JSTensor(ctx, tensor->match_text(ctx.image(), text));
}

template <typename JSTensor>
Value match_foreground(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	typename JSTensor::Opaque* tensor = (typename JSTensor::Opaque*)this_val.get_opaque(JSTensor::class_id);
	if (args.size() != 1) {
		throw std::runtime_error("Invalid arguments count in Tensor::match_foreground");
	}
	std::string color = args.at(0);
	return JSTensor(ctx, tensor->match_color(ctx.image(), color, {}));
}

template <typename JSTensor>
Value match_background(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	typename JSTensor::Opaque* tensor = (typename JSTensor::Opaque*)this_val.get_opaque(JSTensor::class_id);
	if (args.size() != 1) {
		throw std::runtime_error("Invalid arguments count in Tensor::match_background");
	}
	std::string color = args.at(0);
	return JSTensor(ctx, tensor->match_color(ctx.image(), {}, color));
}

template <typename JSTensor>
Value match_text(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	typename JSTensor::Opaque* tensor = (typename JSTensor::Opaque*)this_val.get_opaque(JSTensor::class_id);
	if (args.size() != 1) {
		throw std::runtime_error("Invalid arguments count in Tensor::match_text");
	}
	std::string text = args.at(0);
	return JSTensor(ctx, tensor->match_text(ctx.image(), text));
}

template <typename JSTensor>
Value match_color(ContextRef ctx, ValueRef this_val, const std::vector<ValueRef>& args) {
	typename JSTensor::Opaque* tensor = (typename JSTensor::Opaque*)this_val.get_opaque(JSTensor::class_id);
	if ((args.size() != 2)) {
		throw std::runtime_error("Invalid arguments count in Tensor::match_color");
	}
	std::string fg;
	if (args.at(0).is_string()) {
		fg = (std::string)args.at(0);
	}
	std::string bg;
	if (args.at(1).is_string()) {
		bg = (std::string)args.at(1);
	}
	return JSTensor(ctx, tensor->match_color(ctx.image(), fg, bg));
}

void TextTensor::register_class(ContextRef ctx) {

	static const std::vector<JSCFunctionListEntry> proto_funcs = {
		Prop("[Symbol.toStringTag]", "TextTensor"),
		Method<size<TextTensor>>("size"),
		GetSet<x<TextTensor>, nullptr>("x"),
		GetSet<y<TextTensor>, nullptr>("y"),
		Method<left_top<TextTensor>>("left_top"),
		Method<left_bottom<TextTensor>>("left_bottom"),
		Method<right_top<TextTensor>>("right_top"),
		Method<right_bottom<TextTensor>>("right_bottom"),
		Method<center<TextTensor>>("center"),
		Method<center_top<TextTensor>>("center_top"),
		Method<center_bottom<TextTensor>>("center_bottom"),
		Method<left_center<TextTensor>>("left_center"),
		Method<right_center<TextTensor>>("right_center"),
		Method<from_top<TextTensor>>("from_top"),
		Method<from_bottom<TextTensor>>("from_bottom"),
		Method<from_left<TextTensor>>("from_left"),
		Method<from_right<TextTensor>>("from_right"),
		// specific

		Method<match<TextTensor>>("match"), //< deprecated
		Method<match_foreground<TextTensor>>("match_foreground"), //< deprecated
		Method<match_background<TextTensor>>("match_background"), //< deprecated

		Method<match_text<TextTensor>>("match_text"),
		Method<match_color<TextTensor>>("match_color"),
	};

	Tensor::register_class(ctx, "TextTensor", proto_funcs);
}

void ImgTensor::register_class(ContextRef ctx) {

	static const std::vector<JSCFunctionListEntry> proto_funcs = {
		Prop("[Symbol.toStringTag]", "ImgTensor"),
		Method<size<ImgTensor>>("size"),
		GetSet<x<ImgTensor>, nullptr>("x"),
		GetSet<y<ImgTensor>, nullptr>("y"),
		Method<left_top<ImgTensor>>("left_top"),
		Method<left_bottom<ImgTensor>>("left_bottom"),
		Method<right_top<ImgTensor>>("right_top"),
		Method<right_bottom<ImgTensor>>("right_bottom"),
		Method<center<ImgTensor>>("center"),
		Method<center_top<ImgTensor>>("center_top"),
		Method<center_bottom<ImgTensor>>("center_bottom"),
		Method<left_center<ImgTensor>>("left_center"),
		Method<right_center<ImgTensor>>("right_center"),
		Method<from_top<ImgTensor>>("from_top"),
		Method<from_bottom<ImgTensor>>("from_bottom"),
		Method<from_left<ImgTensor>>("from_left"),
		Method<from_right<ImgTensor>>("from_right"),
	};

	Tensor::register_class(ctx, "ImgTensor", proto_funcs);
}

}
