
#include "OnnxRuntime.hpp"

#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

namespace nn {

std::unique_ptr<Ort::Env> env;

OnnxRuntime::OnnxRuntime() {
	env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "testo");
}

OnnxRuntime::~OnnxRuntime() {
	env.reset();
}

fs::path GetModelDir() {
	return "/usr/share/testo";
}

std::unique_ptr<Ort::Session> LoadModel(const std::string& name) {
	if (!env) {
		throw std::runtime_error("Init onnx runtime first!");
	}
	Ort::SessionOptions session_options;
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	session_options.SetIntraOpNumThreads(1);
	session_options.SetInterOpNumThreads(1);
	session_options.SetExecutionMode(ORT_SEQUENTIAL);
	fs::path model_path = GetModelDir() / (name + ".onnx");
	return std::make_unique<Ort::Session>(*env, 
#ifdef WIN32
		model_path.generic_wstring().c_str(), 
#else
		model_path.generic_string().c_str(), 
#endif
		session_options);
}

}
