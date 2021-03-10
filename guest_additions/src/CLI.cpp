
#include <coro/Application.h>
#include <coro/StreamSocket.h>
#include <clipp.h>
#include <iostream>
#include <guest_additions_common_stuff/GuestAdditions.hpp>

#ifdef __linux__
struct GA: GuestAdditions {
	GA() {
		socket.connect("/var/run/testo-guest-additions.sock");
	}

private:
	void send_raw(const uint8_t* data, size_t size) override {
		size_t n = socket.write(data, size);
		if (n != size) {
			throw std::runtime_error(__PRETTY_FUNCTION__);
		}
	}
	void recv_raw(uint8_t* data, size_t size) override {
		size_t n = socket.read(data, size);
		if (n != size) {
			throw std::runtime_error(__PRETTY_FUNCTION__);
		}
	}

	coro::StreamSocket<asio::local::stream_protocol> socket;
};
#endif

struct MountArgs {
	std::string folder_name;
	std::string guest_path;
	bool permanent = false;
};

struct UmountArgs {
	std::string folder_name;
	bool permanent = false;
};

void mount_mode(const MountArgs& args) {
#if defined (__QEMU__) && defined(__linux__)
	bool was_indeed_mounted = GA().mount(args.folder_name, fs::absolute(args.guest_path), args.permanent);
	if (!was_indeed_mounted) {
		std::cout << "The shared folder is already mounted" << std::endl;
	}
#else
	throw std::runtime_error("Sorry, shared folders are not supported on this combination of the hypervisor and the operating system");
#endif
}

void umount_mode(const UmountArgs& args) {
#if defined (__QEMU__) && defined(__linux__)
	bool was_indeed_umounted = GA().umount(args.folder_name, args.permanent);
	if (!was_indeed_umounted) {
		std::cout << "The shared folder is already umounted" << std::endl;
	}
#else
	throw std::runtime_error("Sorry, shared folders are not supported on this combination of the hypervisor and the operating system");
#endif
}

enum class mode {
	mount,
	umount,
};

mode selected_mode;

int do_main(int argc, char** argv) {

	using namespace clipp;

	mode selected_mode;

	MountArgs mount_args;
	auto mount_spec = "mount options:" % (
		command("mount").set(selected_mode, mode::mount),
		value("folder_name", mount_args.folder_name) % "Shared folder name",
		value("guest_path", mount_args.guest_path) % "Path on the guest where to mount",
		option("--permanent").set(mount_args.permanent) % "Mount this folder automatically after system reboot"
	);

	UmountArgs umount_args;
	auto umount_spec = "umount options:" % (
		command("umount").set(selected_mode, mode::umount),
		value("folder_name", umount_args.folder_name) % "Shared folder name",
		option("--permanent").set(mount_args.permanent) % "Stop mounting this folder automatically after system reboot"
	);

	auto cli = (mount_spec | umount_spec);

	if (!parse(argc, argv, cli)) {
		std::cout << make_man_page(cli, argv[0]) << std::endl;
		return 1;
	}

	switch (selected_mode) {
		case mode::mount:
			mount_mode(mount_args);
			break;
		case mode::umount:
			umount_mode(umount_args);
			break;
		default:
			throw std::runtime_error("Invalid mode");
	}

	return 0;
}

int main(int argc, char** argv) {
	int result = 0;

	coro::Application([&]{
		try {
			result = do_main(argc, argv);
		} catch (const std::exception& error) {
			std::cerr << error.what() << std::endl;
			result = 1;
		}
	}).run();

	return result;
}
