
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>

#include <signal.h>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <coro/Application.h>
#include <coro/CoroPool.h>
#include <coro/Timer.h>
#include <coro/StreamSocket.h>
#include <coro/Acceptor.h>

#include <clipp.h>

#include "MessageHandler.hpp"
#ifdef __HYPERV__
#include "HyperVChannel.hpp"
#elif __QEMU__
#include "QemuLinuxChannel.hpp"
#else
#error "Unknown hypervisor"
#endif

using namespace std::chrono_literals;

#define APP_NAME "testo-guest-additions"
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

void remote_handler(HostMessageHandler& message_handler) {
#ifdef __QEMU__
	std::shared_ptr<Channel> channel(new QemuLinuxChannel);
	coro::Timer timer;
	while (true) {
		try {
			message_handler.run(channel);
		} catch (const std::exception& error) {
			spdlog::error("Error inside QemuLinuxChannel loop: {}", error.what());
			timer.waitFor(100ms);
		}
	}
#elif __HYPERV__
	coro::Acceptor<hyperv::VSocketProtocol> acceptor(hyperv::VSocketEndpoint(HYPERV_PORT));
	while (true) {
		coro::StreamSocket<hyperv::VSocketProtocol> socket = acceptor.accept();
		try {
			message_handler.run(std::make_shared<HyperVChannel>(std::move(socket)));
		} catch (const std::exception& error) {
			spdlog::error("Error inside remote acceptor loop: {}", error.what());
		}
	}
#else
#error "Unknown hypervisor"
#endif
}

struct LocalChannel: Channel {
	using Socket = coro::StreamSocket<asio::local::stream_protocol>;

	LocalChannel(Socket socket_): socket(std::move(socket_)) {}
	~LocalChannel() = default;

	LocalChannel(LocalChannel&& other);
	LocalChannel& operator=(LocalChannel&& other);

	size_t read(uint8_t* data, size_t size) override {
		return socket.readSome(data, size);
	}

	size_t write(uint8_t* data, size_t size) override {
		return socket.writeSome(data, size);
	}

private:
	Socket socket;
};

void local_handler(CLIMessageHandler& message_handler) {
	coro::Acceptor<asio::local::stream_protocol> acceptor("/var/run/testo-guest-additions.sock");
	while (true) {
		coro::StreamSocket<asio::local::stream_protocol> socket = acceptor.accept();
		try {
			message_handler.run(std::make_shared<LocalChannel>(std::move(socket)));
		} catch (const std::exception& error) {
			spdlog::error("Error inside local acceptor loop: {}", error.what());
		}
	}
}

std::thread run_async(std::function<void()> fn_) {
	return std::thread([fn = std::move(fn_)] {
		coro::Application(fn).run();
	});
}

void run_handlers() {
	HostMessageHandler host_handler;
	CLIMessageHandler cli_handler(&host_handler);

	std::thread t1 = run_async([&] { remote_handler(host_handler); });
	std::thread t2 = run_async([&] { local_handler(cli_handler); });

	t1.join();
	t2.join();
}

void app_main() {
	try {
		mount_permanent_shared_folders();
		run_handlers();
	} catch (const std::exception& err) {
		spdlog::error("app_main std error: {}", err.what());
	} catch (const coro::CancelError&) {
		spdlog::error("app_main CancelError");
	} catch (...) {
		spdlog::error("app_main unknown error");
	}
}

void start() {
	if (daemon(1, 0) < 0) {
		throw std::system_error(errno, std::system_category());
	}
	spdlog::info("Starting ...");
	app_main();
	spdlog::info("Stopped");
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

mode selected_mode;

int main(int argc, char** argv) {

	mkdir("/var/log", 0755);

	auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(LOG_FILE_PATH);
	auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
	auto logger = std::make_shared<spdlog::logger>("basic_logger", spdlog::sinks_init_list{file_sink, console_sink});
	logger->set_level(spdlog::level::info);
	logger->flush_on(spdlog::level::info);
	spdlog::set_default_logger(logger);

	try {
		using namespace clipp;

		auto cli = (
			command("start").set(selected_mode, mode::start) |
			command("stop").set(selected_mode, mode::stop) |
			command("status").set(selected_mode, mode::status) |
			command("help").set(selected_mode, mode::help)
		);

		if (!parse(argc, argv, cli)) {
			std::cout << make_man_page(cli, APP_NAME) << std::endl;
			return -1;
		}

		switch (selected_mode) {
			case mode::start:
				start();
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
			case mode::help:
				std::cout << make_man_page(cli, APP_NAME) << std::endl;
				return 0;
			default:
				throw std::runtime_error("Unknown mode");
		}
	}
	catch (const std::exception& error) {
		spdlog::error(error.what());
		return -1;
	}
}