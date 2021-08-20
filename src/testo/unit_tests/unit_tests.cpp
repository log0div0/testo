
#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

int main( int argc, char* argv[] ) {
#ifdef WIN32
	SetConsoleOutputCP(CP_UTF8);
#endif
	int result = Catch::Session().run( argc, argv );
	return result;
}
