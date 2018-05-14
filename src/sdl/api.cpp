
#include "api.hpp"
#include <stdexcept>

namespace sdl {

API::API(uint32_t flags) {
	try {
		if (SDL_Init(flags) != 0) {
			throw std::runtime_error(SDL_GetError());
		}
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

API::~API() {
	SDL_Quit();
}

}
