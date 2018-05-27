
#include "vm.hpp"

VM::VM(): window(sdl::Window(
	"testo",
	SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
	500, 500,
	SDL_WINDOW_SHOWN
)) {

}