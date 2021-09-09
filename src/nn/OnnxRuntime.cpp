
#include "OnnxRuntime.hpp"
#include "TextTensor.hpp"
#include "SelfTestImg.hpp"
#ifdef WIN32
#include "winapi/RegKey.hpp"
#endif

#include <ghc/filesystem.hpp>
namespace fs = ghc::filesystem;

namespace nn {
namespace onnx {

std::unique_ptr<Ort::Env> env;
bool use_cpu = false;

Runtime::Runtime(
#ifdef USE_CUDA
	bool use_cpu_
#endif
) {
	env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "testo");
#ifdef USE_CUDA
	use_cpu = use_cpu_;
#endif
}

Runtime::~Runtime() {
	env.reset();
}

void Runtime::selftest() {
	stb::Image<stb::RGB> img(SelfTestImg, SelfTestImg_len);
	nn::TextTensor tensor = find_text(&img);
	if (tensor.match_text(&img, "Добро пожаловать").size() != 1) {
		throw std::runtime_error("Neural networks are not working correctly");
	}
}

#ifdef __linux__
fs::path GetModelDir() {
	return "/usr/share/testo";
}
#endif
#ifdef WIN32
fs::path GetModelDir() {
	winapi::RegKey regkey(HKEY_LOCAL_MACHINE, "SOFTWARE\\Testo Lang\\Testo NN Service", KEY_QUERY_VALUE);
	return fs::path(regkey.get_str("InstallDir")) / "share";
}
#endif
#ifdef __APPLE__
fs::path GetModelDir() {
	throw std::runtime_error(__PRETTY_FUNCTION__);
}
#endif

Model::Model(const char* name) {
	if (!env) {
		throw std::runtime_error("Init onnx runtime first!");
	}
	Ort::SessionOptions session_options;
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	session_options.SetIntraOpNumThreads(1);
	session_options.SetInterOpNumThreads(1);
	session_options.SetExecutionMode(ORT_SEQUENTIAL);
#ifdef USE_CUDA
	if (!use_cpu) {
		Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
	}
#endif
	fs::path model_path = GetModelDir() / (std::string(name) + ".onnx");
	session = std::make_unique<Ort::Session>(*env,
#ifdef WIN32
		model_path.wstring().c_str(),
#else
		model_path.string().c_str(),
#endif
		session_options);
}

void Model::run(std::initializer_list<Value*> in, std::initializer_list<Value*> out) {
	std::vector<const char*> in_names;
	std::vector<const char*> out_names;
	std::vector<Ort::Value> in_tensors;
	std::vector<Ort::Value> out_tensors;

	for (auto x: in) {
		in_names.push_back(x->name());
		in_tensors.push_back(x->tensor());
	}
	for (auto x: out) {
		out_names.push_back(x->name());
		out_tensors.push_back(x->tensor());
	}

	session->Run(Ort::RunOptions{nullptr},
		in_names.data(), in_tensors.data(), in.size(),
		out_names.data(), out_tensors.data(), out.size());
}

Ort::Value Value::tensor() {
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	return Ort::Value::CreateTensor<float>(memory_info, _buf.data(), _buf.size(), _shape.data(), _shape.size());
}

void Image::set(const stb::Image<stb::RGB>& img, bool normalize, int x_off, int y_off) {
	if (_shape.at(1) != img.c) {
		throw std::runtime_error("_shape.at(1) != img.c");
	}
	if (_shape.at(2) < (y_off + img.h)) {
		throw std::runtime_error("_shape.at(2) < (y_off + img.h)");
	}
	if (_shape.at(3) < (x_off + img.w)) {
		throw std::runtime_error("_shape.at(3) < (x_off + img.w)");
	}
	if (normalize) {
		float mean[3] = {0.485f, 0.456f, 0.406f};
		float std[3] = {0.229f, 0.224f, 0.225f};

		for (int y = 0; y < img.h; ++y) {
			for (int x = 0; x < img.w; ++x) {
				for (int c = 0; c < img.c; ++c) {
					at(x_off + x, y_off + y, c) = ((float(img.at(x, y)[c]) / 255.0f) - mean[c]) / std[c];
				}
			}
		}
	} else {
		for (int y = 0; y < img.h; ++y) {
			for (int x = 0; x < img.w; ++x) {
				for (int c = 0; c < img.c; ++c) {
					at(x_off + x, y_off + y, c) = float(img.at(x, y)[c]) / 255.0f;
				}
			}
		}
	}
}

}
}
