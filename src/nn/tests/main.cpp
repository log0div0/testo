
#include <nn/OnnxRuntime.hpp>
#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

int main( int argc, char* const argv[] ) {
	nn::onnx::Runtime runtime;
	return Catch::Session().run(argc, argv);
}
