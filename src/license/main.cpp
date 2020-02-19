
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

enum Mode {GEN_KEYS, SIGN, VERIFY};
Mode mode;
std::string in_path, out_path;
std::string private_key_path, public_key_path;

void gen_keys() {
	uint8_t public_key[crypto_sign_PUBLICKEYBYTES] = {};
	uint8_t private_key[crypto_sign_SECRETKEYBYTES] = {};
	int result = crypto_sign_keypair(public_key, private_key);
	if (result) {
		throw std::runtime_error("crypto_sign_keypair failed");
	}
	write_file(public_key_path, base64_encode(public_key, crypto_sign_PUBLICKEYBYTES));
	write_file(private_key_path, base64_encode(private_key, crypto_sign_SECRETKEYBYTES));
}

void sign() {
	sign_license(in_path, out_path, read_file(private_key_path));
}

void verify() {
	verify_license(in_path, read_file(public_key_path));
}

int main(int argc, char** argv) {
	try {
		using namespace clipp;

		auto gen_keys_spec = (
			command("gen_keys").set(mode, GEN_KEYS),
			required("--private_key") & value("path", private_key_path),
			required("--public_key") & value("path", public_key_path)
		);

		auto sign_spec = (
			command("sign").set(mode, SIGN),
			required("--in") & value("path", in_path),
			required("--out") & value("path", out_path),
			required("--private_key") & value("path", private_key_path)
		);

		auto verify_spec = (
			command("verify").set(mode, VERIFY),
			required("--in") & value("path", in_path),
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
