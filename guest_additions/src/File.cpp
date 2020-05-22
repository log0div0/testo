
#include "File.hpp"

#ifdef WIN32

#include "winapi.hpp"

void write_file(const std::experimental::filesystem::path& path, const std::vector<uint8_t>& data) {
	winapi::File file(path.string(), GENERIC_WRITE, CREATE_ALWAYS);
	file.write((uint8_t*)&data[0], data.size());
}

std::vector<uint8_t> read_file(const std::experimental::filesystem::path& path) {
	winapi::File file(path.string(), GENERIC_READ, OPEN_EXISTING);
	std::vector<uint8_t> result;
	result.resize(file.size());
	file.read((uint8_t*)&result[0], result.size());
	return result;
}

#else

#include <iterator>
#include <fstream>

std::vector<uint8_t> read_file(const std::experimental::filesystem::path& path) {
	std::ifstream stream(path, std::ios::binary);
	std::noskipws(stream);
	return {std::istream_iterator<uint8_t>(stream), std::istream_iterator<uint8_t>()};
}

void write_file(const std::experimental::filesystem::path& path, const std::vector<uint8_t>& data) {
	std::ofstream stream(path, std::ios::out | std::ios::binary);
	if (!stream) {
		throw std::runtime_error("Couldn't open file stream to write file " + path.string());
	}
	stream.write((const char*)&data[0], data.size());
}

#endif
