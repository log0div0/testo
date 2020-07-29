
#include "License.hpp"
#include <base64.hpp>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

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

std::string pack(const nlohmann::json& j, const std::string& private_key_base64) {
	std::vector<uint8_t> private_key = base64_decode(private_key_base64);
	if (private_key.size() != crypto_sign_SECRETKEYBYTES) {
		throw std::runtime_error("Invalid size of private key");
	}

	std::string message = j.dump();
	std::vector<uint8_t> sig(crypto_sign_BYTES + message.size());
	long long unsigned sig_size = 0;
	int result = crypto_sign(sig.data(), &sig_size, (uint8_t*)message.data(), message.size(), private_key.data());
	if (result) {
		throw std::runtime_error("crypto_sign failed");
	}

	return base64_encode(sig.data(), sig.size());
}

nlohmann::json unpack(const std::string& container, const std::string& public_key_base64) {
	std::vector<uint8_t> sig = base64_decode(container);

	std::vector<uint8_t> public_key = base64_decode(public_key_base64);
	if (public_key.size() != crypto_sign_PUBLICKEYBYTES) {
		throw std::runtime_error("Invalid size of public key");
	}

	std::string message;
	message.resize(sig.size());
	long long unsigned message_size = 0;
	int result = crypto_sign_open((uint8_t*)message.data(), &message_size, (uint8_t*)sig.data(), sig.size(), public_key.data());
	if (result) {
		throw std::runtime_error("The container's digital signature is invalid");
	}
	message.resize(message_size);

	return nlohmann::json::parse(message);
}

}
