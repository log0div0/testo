
#include "License.hpp"
#include <nlohmann/json.hpp>
#include <base64.hpp>
#include <sodium.h>
#include <fstream>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

std::string normalize_license(const nlohmann::json& license) {
	uint64_t id = license.at("id");
	std::string not_before = license.at("not_before").get<std::string>();
	std::string not_after = license.at("not_after").get<std::string>();
	return std::to_string(id) + not_before + not_after;
}

nlohmann::json load_license(const std::string& path) {
	if (!fs::exists(path)) {
		throw std::runtime_error("License file " + path + " does not exists");
	}
	nlohmann::json license;
	try {
		std::ifstream license_file(path);
		license_file >> license;
	} catch (const std::exception&) {
		throw std::runtime_error("Invalid license file format");
	}
	return license;
}

void sign_license(const std::string& license_path, const std::string& private_key_base64) {
	nlohmann::json license = load_license(license_path);

	{
		std::ifstream license_file(license_path);
		license_file >> license;
	}

	std::vector<uint8_t> private_key = base64_decode(private_key_base64);
	if (private_key.size() != crypto_sign_SECRETKEYBYTES) {
		throw std::runtime_error("Invalid size of private key");
	}

	std::string message = normalize_license(license);
	uint8_t sig[crypto_sign_BYTES] = {};
	int result = crypto_sign_detached(sig, nullptr, (uint8_t*)message.data(), message.size(), private_key.data());
	if (result) {
		throw std::runtime_error("crypto_sign_detached failed");
	}

	license["sig"] = base64_encode(sig, crypto_sign_BYTES);

	{
		std::ofstream license_file(license_path);
		license_file << license.dump(4);
	}
}

std::string verify_license(const std::string& license_path, const std::string& public_key_base64) {
	nlohmann::json license = load_license(license_path);

	std::vector<uint8_t> public_key = base64_decode(public_key_base64);
	if (public_key.size() != crypto_sign_PUBLICKEYBYTES) {
		throw std::runtime_error("Invalid size of public key");
	}

	std::vector<uint8_t> sig = base64_decode(license.at("sig"));
	if (sig.size() != crypto_sign_BYTES) {
		throw std::runtime_error("Invalid sig length");
	}

	std::string message = normalize_license(license);

	int result = crypto_sign_verify_detached(sig.data(), (uint8_t*)message.data(), message.size(), public_key.data());
	if (result) {
		throw std::runtime_error("crypto_sign_verify_detached failed");
	}

	auto not_before = Date::from_string(license.at("not_before").get<std::string>()).to_time_point();
	auto not_after = Date::from_string(license.at("not_after").get<std::string>()).to_time_point();
	auto now = std::chrono::system_clock::now();

	if (now < not_before) {
		return "Период действия лицензии ещё не настал";
	}

	if (now > not_after) {
		return "Период действия лицензии ещё не закончился";
	}

	return {};
}
