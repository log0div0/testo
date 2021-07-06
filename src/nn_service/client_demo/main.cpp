#include <iostream>
#include <stdexcept>
#include <thread>
#include <chrono>

#include <clipp.h>

#include <coro/Application.h>
#include <coro/StreamSocket.h>

#include "../Channel.hpp"
#include "../../nn/TextTensor.hpp"
#include "../../nn/ImgTensor.hpp"

using namespace std::chrono_literals;

void draw_rect(stb::Image<stb::RGB>& img, nn::Rect bbox, stb::RGB color) {
	for (int y = bbox.top; y <= bbox.bottom; ++y) {
		img.at(bbox.left, y) = color;
		img.at(bbox.right, y) = color;
	}
	for (int x = bbox.left; x < bbox.right; ++x) {
		img.at(x, bbox.top) = color;
		img.at(x, bbox.bottom) = color;
	}
}

struct Args {
	virtual ~Args() = default;
	std::string ip_port;
	std::string img_file;
};

struct TextArgs: Args {
	std::string query;
	std::string fg;
	std::string bg;
};

struct ImgArgs: Args {
	std::string ref_file;
};

enum class mode {
	text,
	img
};

void text_mode(const TextArgs& args, std::shared_ptr<Channel> channel) {
	auto image = stb::Image<stb::RGB>(args.img_file);

	TextRequest msg(image, args.query, args.fg, args.bg);

	channel->send_request(msg);
	auto response = channel->receive_response();

	std::cout << "Response: " << std::endl;
	std::cout << response.dump(4);
	auto tensor = response.get<nn::TextTensor>();

	for (auto& textline: tensor.objects) {
		draw_rect(image, textline.rect, {200, 20, 50});
	}

	image.write_png("output.png");
}

void img_mode(const ImgArgs& args, std::shared_ptr<Channel> channel) {
	auto image = stb::Image<stb::RGB>(args.img_file);
	auto ref = stb::Image<stb::RGB>(args.ref_file);

	ImgRequest msg(image, ref);

	channel->send_request(msg);
	auto response = channel->receive_response();

	std::cout << "Response: " << std::endl;
	std::cout << response.dump(4);
	auto tensor = response.get<nn::ImgTensor>();

	for (auto& img: tensor.objects) {
		draw_rect(image, img.rect, {200, 20, 50});
	}

	image.write_png("output.png");
}

void handler(const Args& args) {
	try {
		auto semicolon_pos = args.ip_port.find(":");
		if (semicolon_pos == std::string::npos) {
			throw std::runtime_error("ip_port string is malformed: no semicolon");
		}
		auto ip = args.ip_port.substr(0, semicolon_pos);
		auto port = args.ip_port.substr(semicolon_pos + 1, args.ip_port.length() - 1);
		std::cout << "Connecting to " << ip << ":" << port << std::endl;
		auto endpoint = Endpoint(asio::ip::address::from_string(ip), std::stoul(port));
		Socket socket;
		socket.connect(endpoint);
		std::cout << "Connected\n";

		std::shared_ptr<Channel> channel(new Channel(std::move(socket)));

		try {
			const TextArgs& text_args = dynamic_cast<const TextArgs&>(args);
			return text_mode(text_args, channel);
		} catch (const std::bad_cast& e) {
			const ImgArgs& img_args = dynamic_cast<const ImgArgs&>(args);
			return img_mode(img_args, channel);
		}
	} catch (const std::exception& error) {
		std::cout << error.what() << std::endl;
	}
}

int main(int argc, char** argv) {
	try {
		using namespace clipp;

		mode selected_mode;

		TextArgs text_args;
		auto text_spec = (
			command("text").set(selected_mode, mode::text),
			value("input image", text_args.img_file),
			required("--nn_service") & value("ip:port of the nn_service", text_args.ip_port),
			option("--query") & value("the text to search for", text_args.query),
			option("--fg") & value("foreground color", text_args.fg),
			option("--bg") & value("background color", text_args.bg)
		);

		ImgArgs img_args;
		auto img_spec = (
			command("img").set(selected_mode, mode::img),
			value("search image", img_args.img_file),
			value("ref image", img_args.ref_file),
			required("--nn_service") & value("ip:port of the nn_service", img_args.ip_port)
		);

		auto cli = (text_spec | img_spec);

		if (!parse(argc, argv, cli)) {
			std::cout << make_man_page(cli, argv[0]) << std::endl;
			return 1;
		}


		if (selected_mode == mode::text) {
			coro::Application([&](){return handler(text_args);}).run();
		} else if (selected_mode == mode::img) {
			coro::Application([&](){return handler(img_args);}).run();
		}
	} catch (const std::exception& error) {
		std::cout << error.what() << std::endl;
	}
	return 0;
}