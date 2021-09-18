
#include "Main.hpp"
#include <clipp.h>

#define APP_NAME "testo_nn_service"
#define PID_FILE_PATH ("/var/run/" APP_NAME ".pid")
#define LOG_FILE_PATH ("/var/log/" APP_NAME ".log")

struct pid_file_t {
	pid_file_t(pid_t pid) {
		std::ofstream stream(PID_FILE_PATH);
		if (!stream) {
			throw std::runtime_error("Creating pid file failure");
		}
		stream << pid << std::endl;
		stream.flush();
	}
	~pid_file_t() {
		unlink(PID_FILE_PATH);
	}
};

pid_t get_pid() {
	std::ifstream stream(PID_FILE_PATH);

	if (!stream) {
		return 0;
	}

	pid_t pid;
	stream >> pid;
	return pid;
}

struct cmdline: std::vector<std::string> {
	cmdline(pid_t pid) {
		std::string filename = "/proc/" + std::to_string(pid) + "/cmdline";
		std::ifstream stream(filename);
		if (!stream) {
			return;
		}
		for (std::string arg; std::getline(stream, arg, '\0'); ) {
			push_back(arg);
		}
	}
};

inline std::ostream& operator<<(std::ostream& stream, const cmdline& cmdline) {
	for (size_t i = 0; i < cmdline.size(); ++i) {
		if (i) {
			stream << " ";
		}
		stream << cmdline[i];
	}
	return stream;
}

bool is_running() {
	auto pid = get_pid();

	if (!pid) {
		return false;
	}

	if (kill(pid, 0) < 0) {
		return false;
	}
	else {
		cmdline cmd(pid);
		try {
			if (cmd.at(0).find(APP_NAME) != std::string::npos) {
				return true;
			} else {
				return false;
			}
		}
		catch (const std::out_of_range& error) {
			return false;
		}
	}
}

enum class mode {
	start,
	stop,
	status,
	help
};

struct StartArgs {
	std::string settings_path = "/etc/testo/nn_service.json";
	bool foreground_mode = false; 
};

void app_main(const std::string& settings_path) {
	try {
		std::ifstream is(settings_path);
		if (!is) {
			throw std::runtime_error(std::string("Can't open settings file: ") + settings_path);
		}
		is >> settings;
		setup_logs(LOG_FILE_PATH);
	} catch (const std::exception& error) {
		std::cout << error.what() << std::endl;
		return;
	}

	try {
		bool use_gpu = settings.value("use_gpu", false);

		if (use_gpu) {
			if (!settings.count("license_path")) {
				throw std::runtime_error("To start the program in GPU mode you must specify the path to the license file (license_path in the settings file)");
			}
#ifdef USE_CUDA
			verify_license(settings.at("license_path").get<std::string>());
#endif
		}

		nn::onnx::Runtime onnx_runtime(!use_gpu);

		spdlog::info("Starting testo nn service");
		spdlog::info("Testo framework version: {}", TESTO_VERSION);
		spdlog::info("GPU mode enabled: {}", use_gpu);
		coro::Application(local_handler).run();
	} catch (const std::exception& error) {
		spdlog::error(error.what());
	}
}

void start(const StartArgs& args) {
	if (!args.foreground_mode) {
		if (daemon(1, 0) < 0) {
			throw std::system_error(errno, std::system_category());
		}
	}
	auto pid  = getpid();
	pid_file_t pid_t(pid); 

	app_main(args.settings_path);
}

void stop() {
	auto pid = get_pid();
	if (!pid) {
		return;
	}

	if (kill(pid, SIGTERM) < 0) {
		throw std::system_error(errno, std::system_category());
	}
}

int main(int argc, char** argv) {
	try {
		using namespace clipp;

		StartArgs start_args;
		mode selected_mode;

		auto start_spec = (
			command("start").set(selected_mode, mode::start),
			(option("--settings_file") & value("path to file", start_args.settings_path)) % "Path to settings file (default is /etc/testo/nn_service.json",
			(option("--foreground").set(start_args.foreground_mode)) % "Run in foreground"
		);

		auto cli = (
			start_spec |
			command("stop").set(selected_mode, mode::stop) |
			command("status").set(selected_mode, mode::status) |
			command("help").set(selected_mode, mode::help)
		);

		if (!parse(argc, argv, cli)) {
			std::cout << make_man_page(cli, argv[0]) << std::endl;
			return 1;
		}

		switch (selected_mode) {
			case mode::start:
				start(start_args);
				return 0;
			case mode::stop:
				stop();
				return 0;
			case mode::status:
				if (is_running()) {
					std::cout << "RUNNING" << std::endl;
					return 1;
				} else {
					std::cout << "STOPPED" << std::endl;
					return 0;
				}
				return 0;
			case mode::help:
				std::cout << make_man_page(cli, APP_NAME) << std::endl;
				return 0;
			default:
				throw std::runtime_error("Unknown mode");
		}
	} catch (const std::exception& error) {
		std::cerr << error.what() << std::endl;
	}
}