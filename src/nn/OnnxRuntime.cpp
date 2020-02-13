
#include "OnnxRuntime.hpp"

namespace nn {

std::unique_ptr<Ort::Env> env;

OnnxRuntime::OnnxRuntime() {
	env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "testo");
}

OnnxRuntime::~OnnxRuntime() {
	env.reset();
}

std::unique_ptr<Ort::Session> LoadModel(unsigned char* data, unsigned int size) {
	if (!env) {
		throw std::runtime_error("Init onnx runtime first!");
	}
	Ort::SessionOptions session_options;
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	session_options.SetIntraOpNumThreads(1);
	session_options.SetInterOpNumThreads(1);
	session_options.SetExecutionMode(ORT_SEQUENTIAL);
	return std::make_unique<Ort::Session>(*env, data, size, session_options);
}

}