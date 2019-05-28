
#include <iostream>
#include <chrono>
#include "Connect.hpp"

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

void main() {
	try {
		wmi::CoInitializer initializer;
		initializer.initalize_security();

		auto start = std::chrono::high_resolution_clock::now();
		msft::Connect connect;
		auto disk = connect.virtualDisk("C:\\Users\\log0div0\\VirtualBox Flash Drives\\images\\pubkey_flash.vhd");
		for (auto& partition: disk.partitions()) {
			partition.deleteObject();
		}
		disk.clear();
		disk.initialize();
		disk.createPartition();
		auto partition = disk.partitions().at(0);
		auto volume = partition.volume();
		volume.format("NTFS", "pubkey_flash");
		for (auto& path: partition.accessPaths()) {
			std::cout << path << std::endl;
		}
		partition.addAccessPath("C:\\Users\\log0div0\\VirtualBox Flash Drives\\mount_point");
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time = end - start;
		std::cout << time.count() << " seconds" << std::endl;

	} catch (const std::exception& error) {
		std::cerr << error << std::endl;
	}
}
