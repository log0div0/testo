
#include <iostream>
#include <clipp.h>
#include <sodium.h>
#include <fstream>
#include <regex>
#include <string>
#include <nlohmann/json.hpp>
#include <base64.hpp>

void backtrace(std::ostream& stream, const std::exception& error, size_t n) {
	stream << n << ". " << error.what();
	try {
		std::rethrow_if_nested(error);
	} catch (const std::exception& error) {
		stream << std::endl;
		backtrace(stream, error, n + 1);
	} catch(...) {
		stream << std::endl;
		stream << n << ". " << "[Unknown exception type]";
	}
}

std::ostream& operator<<(std::ostream& stream, const std::exception& error) {
	backtrace(stream, error, 1);
	return stream;
}

struct Date {
	static Date from_string(const std::string& str) {
		std::regex regex(R"((\d+).(\d+).(\d+))");
		std::cmatch match;
		if (!std::regex_match(str.data(), match, regex)) {
			throw std::runtime_error("Invalid date format");
		}
		Date date;
		date.day = stoi(match[1]);
		date.month = stoi(match[2]);
		date.year = stoi(match[3]);

		if (date.month > 12) {
			throw std::runtime_error("Invalid month number");
		}
		if (date.day > 31) {
			throw std::runtime_error("Invalid day number");
		}

		return date;
	}

	uint16_t day = 0, month = 0, year = 0;
};

std::vector<uint8_t> read_key(const std::string& path) {
	std::ifstream file(path);
	std::string base64;
	file >> base64;
	return base64_decode(base64);
}

void write_key(const std::string& path, const uint8_t* data, size_t size) {
	std::ofstream file(path);
	file << base64_encode(data, size);
}

enum Mode {GEN_KEYS, SIGN, VERIFY};
Mode mode;
std::string license_path;
std::string private_key_path, public_key_path;

void gen_keys() {
	uint8_t public_key[crypto_sign_PUBLICKEYBYTES] = {};
	uint8_t private_key[crypto_sign_SECRETKEYBYTES] = {};
	int result = crypto_sign_keypair(public_key, private_key);
	if (result) {
		throw std::runtime_error("crypto_sign_keypair failed");
	}
	write_key(public_key_path, public_key, crypto_sign_PUBLICKEYBYTES);
	write_key(private_key_path, private_key, crypto_sign_SECRETKEYBYTES);
}

std::string normalize_license(const nlohmann::json& license) {
	uint64_t id = license.at("id");
	std::string not_before = license.at("not_before").get<std::string>();
	std::string not_after = license.at("not_after").get<std::string>();
	Date::from_string(not_before);
	Date::from_string(not_after);
	return std::to_string(id) + not_before + not_after;
}

void sign() {
	nlohmann::json license;

	{
		std::ifstream license_file(license_path);
		license_file >> license;
	}

	std::vector<uint8_t> private_key = read_key(private_key_path);
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

void verify() {
	nlohmann::json license;

	std::ifstream license_file(license_path);
	license_file >> license;

	std::vector<uint8_t> public_key = read_key(public_key_path);
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

	std::cout << "sig is ok" << std::endl;
}

int main(int argc, char** argv) {
	try {
		if (sodium_init() == -1) {
        throw std::runtime_error("sodium_init failed");
    }

		using namespace clipp;

		auto gen_keys_spec = (
			command("gen_keys").set(mode, GEN_KEYS),
			required("--private_key") & value("path", private_key_path),
			required("--public_key") & value("path", public_key_path)
		);

		auto sign_spec = (
			command("sign").set(mode, SIGN),
			required("--license") & value("path", license_path),
			required("--private_key") & value("path", private_key_path)
		);

		auto verify_spec = (
			command("verify").set(mode, VERIFY),
			required("--license") & value("path", license_path),
			required("--public_key") & value("path", public_key_path)
		);

		auto cli = ( gen_keys_spec | sign_spec | verify_spec );

		if (!parse(argc, argv, cli)) {
			std::cout << make_man_page(cli, "license_demo") << std::endl;
			return -1;
		}

		switch (mode) {
			case GEN_KEYS:
				gen_keys();
				break;
			case SIGN:
				sign();
				break;
			case VERIFY:
				verify();
				break;
			default:
				throw std::runtime_error("Should not be here");
		}

	} catch (const std::exception& error) {
		std::cerr << error << std::endl;
	}
	return 0;
}
