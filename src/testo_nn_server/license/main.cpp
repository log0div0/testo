
#include <iostream>
#include <clipp.h>
#include <base64.hpp>
extern "C" {
#include <tweetnacl/tweetnacl.h>
}
#include "License.hpp"

enum Mode {
	GEN_KEYS,
	ISSUE,
	DUMP_LICENSE,
	DUMP_LICENSE_REQUEST
};

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
	std::string request_base64;
	std::cin >> request_base64;
	nlohmann::json license_request = license::unpack(request_base64, "K4fDIgPMK/F/CFouJm4b4y0S60vECLOhsNGYkpkFyAQ=");
	nlohmann::json license = nlohmann::json::object();
	license["device_uuid"] = license_request.at("device_uuid");
	std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
	license::Date not_before(now);
	license["not_before"] = not_before.to_string();
	if (license_type == "full") {
		license::Date not_after(now + std::chrono::hours(24 * 365));
		license["not_after"] = not_after.to_string();
	} else if (license_type == "demo") {
		license::Date not_after(now + std::chrono::hours(24 * 14));
		license["not_after"] = not_after.to_string();
	} else {
		throw std::runtime_error("Invalid license type: " + license_type);
	}
	license["version"] = 1;
	std::cout << license::pack(license, "Z8Zpc1H/Suwpzbqr8vvjRnzCVPgb6OeNdlouYzOfyZqvzVNEO3kNKu9Fnci+vD2mIk/7kqqAGUxfDmMo4+RJJw==");
}

void dump_license() {
	std::string base64;
	std::cin >> base64;
	nlohmann::json j = license::unpack(base64, "r81TRDt5DSrvRZ3Ivrw9piJP+5KqgBlMXw5jKOPkSSc=");
	std::cout << j.dump();
}

void dump_license_request() {
	std::string base64;
	std::cin >> base64;
	nlohmann::json j = license::unpack(base64, "K4fDIgPMK/F/CFouJm4b4y0S60vECLOhsNGYkpkFyAQ=");
	std::cout << j.dump();
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

		auto dump_license_spec = (
			command("dump_license").set(mode, DUMP_LICENSE)
		);

		auto dump_license_request_spec = (
			command("dump_license_request").set(mode, DUMP_LICENSE_REQUEST)
		);

		auto cli = (
			gen_keys_spec |
			issue_spec |
			dump_license_spec |
			dump_license_request_spec
		);

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
			case DUMP_LICENSE:
				dump_license();
				break;
			case DUMP_LICENSE_REQUEST:
				dump_license_request();
				break;
			default:
				throw std::runtime_error("Should not be here");
		}
		return 0;
	} catch (const std::exception& error) {
		std::cerr << error.what() << std::endl;
		return 1;
	}
}
