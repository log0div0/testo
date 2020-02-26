
#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include <onnxruntime_cxx_api.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

namespace nn {

struct OnnxRuntime {
	OnnxRuntime();
	~OnnxRuntime();

	OnnxRuntime(const OnnxRuntime&) = delete;
	OnnxRuntime& operator=(const OnnxRuntime&) = delete;
	OnnxRuntime(OnnxRuntime&&) = delete;
	OnnxRuntime& operator=(OnnxRuntime&&) = delete;
};

std::unique_ptr<Ort::Session> LoadModel(const std::string& name);

}
