
#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <onnxruntime_cxx_api.h>
#pragma GCC diagnostic pop

namespace nn {

struct OnnxRuntime {
	OnnxRuntime();
	~OnnxRuntime();

	OnnxRuntime(const OnnxRuntime&) = delete;
	OnnxRuntime& operator=(const OnnxRuntime&) = delete;
	OnnxRuntime(OnnxRuntime&&) = delete;
	OnnxRuntime& operator=(OnnxRuntime&&) = delete;
};

std::unique_ptr<Ort::Session> LoadModel(unsigned char* data, unsigned int size);

}
