
#pragma once

#include <vector>
#include <stb/Image.hpp>
#include "Homm3Object.hpp"
#include "OnnxRuntime.hpp"

namespace nn {

struct Homm3Detector {
	static Homm3Detector& instance();

	Homm3Detector(const Homm3Detector& root) = delete;
	Homm3Detector& operator=(const Homm3Detector&) = delete;

	std::vector<Homm3Object> detect(const stb::Image<stb::RGB>* image);

private:
	Homm3Detector() = default;
	void run_nn(const stb::Image<stb::RGB>* image);
	std::vector<Homm3Object> run_postprocessing();

	int in_w = 0;
	int in_h = 0;
	int pred_count = 0;
	int pred_length = 0;

	onnx::Model model = "Homm3Detector";
	onnx::Image in = "input";

	struct Output: onnx::Value {
		using onnx::Value::Value;

		void resize(int n, int c) {
			Value::resize({n, c});
		}
		float* operator[](int x) {
			return &_buf[x * _shape[1]];
		}
	};

	Output out = "output";
};

}
