
#include <iostream>
#include <thread>
#include <fstream>
#include "Runtime.hpp"
#include <nn/Context.hpp>
#include <nn/OnnxRuntime.hpp>
#include <clipp.h>

void backtrace(std::ostream& stream, const std::exception& error, size_t n) {
	stream << n << ". " << error.what();
	try {
		std::rethrow_if_nested(error);
	} catch (const std::exception& error) {
		stream << std::endl;
		backtrace(stream, error, n + 1);
	} catch(...) {
		stream << std::endl;
		stream << n << ". " << "[Unknown exception type]";
	}
}

std::ostream& operator<<(std::ostream& stream, const std::exception& error) {
	backtrace(stream, error, 1);
	return stream;
}

int main(int argc, char** argv) {
	try {
		using namespace clipp;

		std::string image_path;
		std::string script_path;

		auto cli = (
			option("--image") & value("image file", image_path),
			option("--script") & value("script file", script_path)
		);

		if (!parse(argc, argv, cli)) {
			std::cout << make_man_page(cli, argv[0]) << std::endl;
			return 1;
		}

		stb::Image image(image_path);
		std::ifstream script_file(script_path);
		if (!script_file.is_open()) {
			throw std::runtime_error("Failed to open script file");
		}
		std::string script = {
			std::istreambuf_iterator<char>(script_file),
			std::istreambuf_iterator<char>()
		};

		quickjs::Runtime js_runtime = quickjs::create_runtime();
		quickjs::Context js_ctx = js_runtime.create_context();
		js_ctx.register_nn_functions();

		nn::OnnxRuntime runtime;
		nn::Context nn_ctx(&image);
		js_ctx.set_opaque(&nn_ctx);

		auto val = js_ctx.eval(script);

		return 0;
	} catch (const std::exception& error) {
		std::cerr << error << std::endl;
	}
}
