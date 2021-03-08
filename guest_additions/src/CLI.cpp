
#include <coro/Application.h>
#include <clipp.h>
#include <iostream>

struct MountArgs {
	std::string folder_name;
	std::string guest_path;
	bool permanent;
};

struct UmountArgs {
	std::string folder_name;
	bool permanent;
};

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
