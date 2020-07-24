
#include "OnnxRuntime.hpp"
#include "OCR.hpp"
#include "SelfTestImg.hpp"
#ifdef WIN32
#include "winapi.hpp"
#endif

#include <experimental/filesystem>

#ifdef USE_CUDA
#include <cuda_provider_factory.h>
#endif

namespace fs = std::experimental::filesystem;

namespace nn {

std::unique_ptr<Ort::Env> env;

OnnxRuntime::OnnxRuntime() {
	env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "testo");
}

OnnxRuntime::~OnnxRuntime() {
	env.reset();
}

#ifdef __linux__
fs::path GetModelDir() {
	return "/usr/share/testo";
}
#endif
#ifdef WIN32
fs::path GetModelDir() {
	winapi::RegKey regkey(HKEY_LOCAL_MACHINE, "SOFTWARE\\Testo Lang\\Testo");
	return regkey.query_str("InstallDir");
}
#endif
#ifdef __APPLE__
fs::path GetModelDir() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
#endif

std::unique_ptr<Ort::Session> LoadModel(const std::string& name) {
	if (!env) {
		throw std::runtime_error("Init onnx runtime first!");
	}
	Ort::SessionOptions session_options;
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	session_options.SetIntraOpNumThreads(1);
	session_options.SetInterOpNumThreads(1);
	session_options.SetExecutionMode(ORT_SEQUENTIAL);
#ifdef USE_CUDA
	Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#endif
	fs::path model_path = GetModelDir() / (name + ".onnx");
	return std::make_unique<Ort::Session>(*env,
#ifdef WIN32
		model_path.wstring().c_str(),
#else
		model_path.string().c_str(),
#endif
		session_options);
}

void OnnxRuntime::selftest() {
	stb::Image img(SelfTestImg, SelfTestImg_len);
	nn::Tensor tensor = find_text(&img);
	if (tensor.match("Добро пожаловать").size() != 1) {
		throw std::runtime_error("Neural networks are not working correctly");
	}
}

}
