
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
	Value(const char* name): _name(name) {};

	const char* name() {
		return _name;
	}

	Ort::Value tensor();

protected:
	void resize(std::vector<int64_t> new_shape) {
		_shape = std::move(new_shape);
		size_t size = 1;
		for (auto dim: _shape) {
			size *= dim;
		}
		_buf.resize(size);
	}

	const char* _name;
	std::vector<float> _buf;
	std::vector<int64_t> _shape;
};

struct Image: Value {
	using Value::Value;

	void resize(int w, int h, int c) {
		Value::resize({1, c, h, w});
		stride_c = h * w;
		stride_y = w;
	}

	void set(const stb::Image<stb::RGB>& img, bool normalize);

	void fill(float value) {
		std::fill(_buf.begin(), _buf.end(), value);
	}

	float& at(int x, int y, int c) {
		return _buf[c*stride_c + y*stride_y + x];
	}

	int stride_c;
	int stride_y;
};

struct Model {
	Model(const char* name);

	void run(std::initializer_list<Value*> in, std::initializer_list<Value*> out);

private:
	std::unique_ptr<Ort::Session> session;
};

#ifdef USE_CUDA
	CUDA_DeviceInfo GetDeviceInfo();
#endif

}
}
