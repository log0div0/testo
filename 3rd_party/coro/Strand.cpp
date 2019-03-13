
#include "coro/Strand.h"

namespace coro {

thread_local Strand* t_strand = nullptr;

Strand* Strand::current() {
	if (!t_strand) {
		throw std::logic_error("Current strand is null");
	}
	return t_strand;
}

}