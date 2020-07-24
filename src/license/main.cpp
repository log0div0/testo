
#include <iostream>
#include <clipp.h>
#include <fstream>
#include <base64.hpp>
extern "C" {
#include <tweetnacl/tweetnacl.h>
}
#include "License.hpp"

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

std::string read_file(const std::string& path) {
	std::ifstream file(path);
	std::string data;
	file >> data;
	return data;
}

void write_file(const std::string& path, const std::string& data) {
	std::ofstream file(path);
	file << data;
}

enum Mode {GEN_KEYS, ISSUE, DUMP_JSON};
Mode mode;

void gen_keys() {
	uint8_t public_key[crypto_sign_PUBLICKEYBYTES] = {};
	uint8_t private_key[crypto_sign_SECRETKEYBYTES] = {};
	int result = crypto_sign_keypair(public_key, private_key);
	if (result) {
		throw std::runtime_error("crypto_sign_keypair failed");
	}
	std::cout << "PUBLIC: " << base64_encode(public_key, crypto_sign_PUBLICKEYBYTES) << std::endl;
	std::cout << "PRIVATE: " << base64_encode(private_key, crypto_sign_SECRETKEYBYTES) << std::endl;
}

void issue(const std::string& license_type) {
	throw std::runtime_error("Implement me");
}

void dump_json() {
	throw std::runtime_error("Implement me");
}

int main(int argc, char** argv) {
	try {
		using namespace clipp;

		auto gen_keys_spec = (
			command("gen_keys").set(mode, GEN_KEYS)
		);

		std::string license_type;

		auto issue_spec = (
			command("issue").set(mode, ISSUE),
			required("--type") & value("license type", license_type)
		);

		auto dump_json_spec = (
			command("dump_json").set(mode, DUMP_JSON)
		);

		auto cli = ( gen_keys_spec | issue_spec | dump_json_spec );

		if (!parse(argc, argv, cli)) {
			std::cout << make_man_page(cli, "testo_license_cgi") << std::endl;
			return -1;
		}

		switch (mode) {
			case GEN_KEYS:
				gen_keys();
				break;
			case ISSUE:
				issue(license_type);
				break;
			case DUMP_JSON:
				dump_json();
				break;
			default:
				throw std::runtime_error("Should not be here");
		}

	} catch (const std::exception& error) {
		std::cerr << error << std::endl;
	}
	return 0;
}
