
#include "api.hpp"
#include <stdexcept>

namespace sdl {

API::API(uint32_t flags) {
	if (SDL_Init(flags) != 0) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

API::~API() {
	SDL_Quit();
}

}
