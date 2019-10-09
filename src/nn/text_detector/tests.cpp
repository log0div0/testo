
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include "TextDetector.hpp"

TEST_CASE("ubuntu18.04 installation") {
	TextDetector detector;
	stb::Image img("1.png");
	CHECK(detector.detect(img, "Install Ubuntu Server").size() == 2);
	CHECK(detector.detect(img, "Uninstall Ubuntu Server").size() == 0);
	CHECK(detector.detect(img, "Install Ubuntu Client").size() == 0);
}
