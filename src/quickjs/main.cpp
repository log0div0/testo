
#include <iostream>
#include "Runtime.hpp"

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

int main() {
	try {
		std::string script = "print('hello world')";
		quickjs::Runtime js_runtime = quickjs::create_runtime();
		quickjs::Context js_ctx = js_runtime.create_context();
		js_ctx.register_nn_functions();
		// nn::Context nn_ctx(&screenshot);
		// js_ctx.set_opaque(&nn_ctx);
		js_ctx.eval(script);
		return 0;
	} catch (const std::exception& error) {
		std::cerr << error << std::endl;
	}
}
