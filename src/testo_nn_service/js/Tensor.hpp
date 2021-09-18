
#pragma once

#include "Context.hpp"
#include "../nn/TextTensor.hpp"
#include "../nn/ImgTensor.hpp"

namespace js {

template <typename JSTensor>
void finalizer(JSRuntime* rt, JSValue val) {
	typename JSTensor::Opaque* tensor = (typename JSTensor::Opaque*)JS_GetOpaque(val, JSTensor::class_id);
	delete tensor;
}

template <typename NNTensor>
struct Tensor: Value {
	static JSClassID class_id;
	static JSClassDef class_def;

	using Opaque = NNTensor;

	static void register_class(ContextRef ctx, const char* name, const std::vector<JSCFunctionListEntry>& proto_funcs) {
		if (!class_id) {
			JS_NewClassID(&class_id);
			class_def.class_name = name;
			class_def.finalizer = finalizer<Tensor>;
			JS_NewClass(JS_GetRuntime(ctx.handle), class_id, &class_def);
		}

		Value proto = ctx.new_object();
		proto.set_property_function_list(proto_funcs.data(), proto_funcs.size());
		ctx.set_class_proto(class_id, std::move(proto));
	}

	Tensor(ContextRef ctx, const NNTensor& tensor): Value(ctx.new_object_class(class_id)) {
		set_opaque(new NNTensor(tensor));
	}
};

template <typename NNTensor>
JSClassID Tensor<NNTensor>::class_id = 0;
template <typename NNTensor>
JSClassDef Tensor<NNTensor>::class_def = {};

struct TextTensor: Tensor<nn::TextTensor> {
	static void register_class(ContextRef ctx);

	using Tensor<nn::TextTensor>::Tensor;
};

struct ImgTensor: Tensor<nn::ImgTensor> {
	static void register_class(ContextRef ctx);

	using Tensor<nn::ImgTensor>::Tensor;
};

}
