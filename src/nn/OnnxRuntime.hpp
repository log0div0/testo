
#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#include <onnxruntime_cxx_api.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#ifdef USE_CUDA
#include <cuda_provider_factory.h>
#endif

#include <stb/Image.hpp>

namespace nn {
namespace onnx {

struct Runtime {
	Runtime();
	~Runtime();

	Runtime(const Runtime&) = delete;
	Runtime& operator=(const Runtime&) = delete;
	Runtime(Runtime&&) = delete;
	Runtime& operator=(Runtime&&) = delete;

	void selftest();
};

struct Value {
	Value(const char* name);
};

struct Image: Value {
	using Value::Value;
	void resize(int w, int h, int c);
	template <typename T>
	void set(const stb::Image<T>& img, bool normalize);
	void fill(float value);
	float* at(int x, int y);
};

struct Model {
	Model(const char* name);

	void run(std::initializer_list<Value*> in, std::initializer_list<Value*> out);
};

#ifdef USE_CUDA
	CUDA_DeviceInfo GetDeviceInfo();
#endif

}
}
