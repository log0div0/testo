
#include "License.hpp"
#include <nlohmann/json.hpp>
#include <base64.hpp>
#include <fstream>

extern "C" {
#include <tweetnacl/tweetnacl.h>
#include <time.h>
void randombytes(uint8_t* buf, uint64_t size) {
	srand(time(0));
	for (uint64_t i = 0; i < size; ++i) {
		buf[i] = rand();
	}
}
}

namespace license {

}
