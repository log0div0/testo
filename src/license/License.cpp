
#include "License.hpp"
#include <nlohmann/json.hpp>
#include <base64.hpp>
#include <fstream>
extern "C" {
#include <tweetnacl/tweetnacl.h>
void randombytes(uint8_t* buf, uint64_t size) {
	srand(0);
	for (uint64_t i = 0; i < size; ++i) {
		buf[i] = rand();
	}
}
}
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

void sign_license(const std::string& in_path, const std::string& out_path, const std::string& private_key_base64) {
	if (!fs::exists(in_path)) {
		throw std::runtime_error("File " + in_path + " does not exists");
	}
	nlohmann::json license_template;
	{
		std::ifstream file(in_path);
		if (!file.is_open()) {
			throw std::runtime_error("Failed to open file " + in_path);
		}
		file >> license_template;
	}

	std::vector<uint8_t> private_key = base64_decode(private_key_base64);
	if (private_key.size() != crypto_sign_SECRETKEYBYTES) {
		throw std::runtime_error("Invalid size of private key");
	}

	std::string message = license_template.dump();
	std::vector<uint8_t> sig(crypto_sign_BYTES + message.size());
	long long unsigned sig_size = 0;
	int result = crypto_sign(sig.data(), &sig_size, (uint8_t*)message.data(), message.size(), private_key.data());
	if (result) {
		throw std::runtime_error("crypto_sign failed");
	}

	{
		std::ofstream file(out_path, std::ios::binary);
		if (!file.is_open()) {
			throw std::runtime_error("Failed to open file " + out_path);
		}
		file.write((char*)sig.data(), sig.size());
	}
}

void verify_license(const std::string& in_path, const std::string& public_key_base64) {
	if (!fs::exists(in_path)) {
		throw std::runtime_error("File " + in_path + " does not exists");
	}

	std::vector<uint8_t> sig;
	{
		std::ifstream file(in_path, std::ios::binary | std::ios::ate);
		if (!file.is_open()) {
			throw std::runtime_error("Failed to open file " + in_path);
		}
		int size = file.tellg();
		file.seekg(0, std::ios::beg);
		sig.resize(size);
		file.read((char*)sig.data(), size);
	}

	std::vector<uint8_t> public_key = base64_decode(public_key_base64);
	if (public_key.size() != crypto_sign_PUBLICKEYBYTES) {
		throw std::runtime_error("Invalid size of public key");
	}

	std::string message;
	message.resize(sig.size());
	long long unsigned message_size = 0;
	int result = crypto_sign_open((uint8_t*)message.data(), &message_size, (uint8_t*)sig.data(), sig.size(), public_key.data());
	if (result) {
		throw std::runtime_error("crypto_sign_open failed");
	}
	message.resize(message_size);

	nlohmann::json license = nlohmann::json::parse(message);

	auto not_before = Date::from_string(license.at("not_before").get<std::string>());
	auto not_after = Date::from_string(license.at("not_after").get<std::string>());
	auto now = Date::now();

	if (now < not_before) {
		throw std::runtime_error("Период действия лицензии ещё не настал");
	}

	if (now > not_after) {
		throw std::runtime_error("Период действия лицензии уже закончился");
	}
}
