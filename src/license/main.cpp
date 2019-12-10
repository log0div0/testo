
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

enum Mode {GEN_KEYS, SIGN, VERIFY};
Mode mode;
uint64_t id = 0;
std::string not_before, not_after;
std::string license_path;
std::string private_key_path, public_key_path;

void gen_keys() {
	uint8_t public_key[crypto_sign_PUBLICKEYBYTES] = {};
	uint8_t private_key[crypto_sign_SECRETKEYBYTES] = {};
	int result = crypto_sign_keypair(public_key, private_key);
	if (result) {
		throw std::runtime_error("crypto_sign_keypair failed");
	}
	std::ofstream public_file(public_key_path, std::ios::binary);
	public_file.write((const char*)public_key, crypto_sign_PUBLICKEYBYTES);
	std::ofstream private_file(private_key_path, std::ios::binary);
	private_file.write((const char*)private_key, crypto_sign_SECRETKEYBYTES);
}

void sign() {
	std::ifstream private_file(private_key_path, std::ios::binary);
	private_file.unsetf(std::ios::skipws);
	std::vector<uint8_t> private_key {
		std::istream_iterator<uint8_t>(private_file),
		std::istream_iterator<uint8_t>()
	};
	if (private_key.size() != crypto_sign_SECRETKEYBYTES) {
		throw std::runtime_error("Invalid size of private key");
	}

	Date::from_string(not_before);
	Date::from_string(not_after);

	std::string message = std::to_string(id) + not_before + not_after;
	uint8_t sig[crypto_sign_BYTES] = {};
	int result = crypto_sign_detached(sig, nullptr, (uint8_t*)message.data(), message.size(), private_key.data());
	if (result) {
		throw std::runtime_error("crypto_sign_detached failed");
	}

	nlohmann::json license = {
		{"id", id},
		{"not_before", not_before},
		{"not_after", not_after},
		{"sig", base64_encode(sig, crypto_sign_BYTES)}
	};

	std::ofstream license_file(license_path);
	license_file << license.dump(4);
}

void verify() {
	std::ifstream license_file(license_path);
	nlohmann::json license;
	license_file >> license;

	std::ifstream public_file(public_key_path, std::ios::binary);
	public_file.unsetf(std::ios::skipws);
	std::vector<uint8_t> public_key {
		std::istream_iterator<uint8_t>(public_file),
		std::istream_iterator<uint8_t>()
	};
	if (public_key.size() != crypto_sign_PUBLICKEYBYTES) {
		throw std::runtime_error("Invalid size of public key");
	}

	id = license.at("id");
	not_before = license.at("not_before").get<std::string>();
	not_after = license.at("not_after").get<std::string>();
	std::vector<uint8_t> sig = base64_decode(license.at("sig"));
	if (sig.size() != crypto_sign_BYTES) {
		throw std::runtime_error("Invalid sig length");
	}

	std::string message = std::to_string(id) + not_before + not_after;

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
			required("--id") & value("id", id),
			required("--not_before") & value("data", not_before),
			required("--not_after") & value("date", not_after),
			required("--private_key") & value("path", private_key_path),
			required("--license") & value("path", license_path)
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
