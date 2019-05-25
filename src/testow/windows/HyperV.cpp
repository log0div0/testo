
#include "HyperV.hpp"
#include <iostream>

HyperVGuest::HyperVGuest(std::string name_): Guest(name_) {

}

uint8_t Table5[1 << 5] = {0, 8, 16, 25, 33, 41, 49, 58, 66, 74, 82, 90, 99, 107, 115, 123, 132,
 140, 148, 156, 165, 173, 181, 189, 197, 206, 214, 222, 230, 239, 247, 255};

uint8_t Table6[1 << 6] = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 45, 49, 53, 57, 61, 65, 69,
 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 130, 134, 138,
 142, 146, 150, 154, 158, 162, 166, 170, 174, 178, 182, 186, 190, 194, 198,
 202, 206, 210, 215, 219, 223, 227, 231, 235, 239, 243, 247, 251, 255};


stb::Image HyperVGuest::screenshot() {
	auto machine = connect.machine(name());
	if (machine.state() != hyperv::Machine::State::Enabled) {
		return {};
	}
	auto display = machine.display();

	size_t height = display.height();
	size_t width = display.width();
	std::vector<uint8_t> screenshot = display.screenshot();

	stb::Image result(width, height, 3);

	for (size_t h = 0; h < height; ++h) {
		for (size_t w = 0; w < width; ++w) {
			size_t dst_index = h*width*3 + w*3;
			size_t src_index = h*width*2 + w*2;
			uint16_t word = *(uint16_t*)(screenshot.data() + src_index);
			uint8_t r5 = word >> 11;
			uint8_t g6 = (word >> 5) & 0b00111111;
			uint8_t b5 = word & 0b00011111;
			uint8_t r8 = Table5[r5];
			uint8_t g8 = Table6[g6];
			uint8_t b8 = Table5[b5];

			result._data[dst_index + 0] = r8;
			result._data[dst_index + 1] = g8;
			result._data[dst_index + 2] = b8;
		}
	}

	return result;
}

HyperV::HyperV() {
	initializer.initalize_security();
}

std::vector<std::shared_ptr<Guest>> HyperV::guests() const {
	std::vector<std::shared_ptr<Guest>> result;
	hyperv::Connect connect;
	for (auto& machine: connect.machines()) {
		result.push_back(std::make_shared<HyperVGuest>(machine.name()));
	}
	return result;
}