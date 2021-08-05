#include <iostream>
#include <stdexcept>
#include <thread>
#include <chrono>

#include <clipp.h>

#include <coro/Application.h>
#include <coro/StreamSocket.h>

#include "Channel.hpp"
#include "Messages.hpp"
#include "../nn/TextTensor.hpp"
#include "../nn/ImgTensor.hpp"

#include <fmt/format.h>

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

struct JSArgs: Args {
	std::string script_file;
};

enum class mode {
	text,
	img,
	js
};

std::string build_js_script_text(std::string query = "", std::string fg = "", std::string bg = "") {
	std::string result = "return ";
	result += query.length() ? fmt::format("find_text('{}')", query) : "find_text()";

	if (fg.length() || bg.length()) {
		result += "match_color(";
		result += fg.length() ? fg : "null";
		result += ", ";
		result += bg.length() ? bg : "null";
		result += ")";
	}

	return result;
}

std::string build_js_script_img(std::string ref_image) {
	std::string result = fmt::format("return find_img('{}')", ref_image);

	return result;
}

void text_mode(const TextArgs& args, std::shared_ptr<Channel> channel) {
	auto image = stb::Image<stb::RGB>(args.img_file);
	auto js_script = build_js_script_text(args.query, args.fg, args.bg);

	std::cout << "Script: " << js_script << std::endl;
	auto request = create_js_eval_request(image, js_script);
	channel->send(request);
	request["image"] = "omitted";
	std::cout << request.dump(4) << std::endl;
	auto response = channel->recv();

	std::cout << "Response: " << std::endl;
	std::cout << response.dump(4) << std::endl << std::endl;

	/*auto data = response.at("data");
	for (auto& textline: data) {
		nn::Rect bbox{
			textline.at("left").get<int32_t>(),
			textline.at("top").get<int32_t>(),
			textline.at("right").get<int32_t>(),
			textline.at("bottom").get<int32_t>()
		};
		draw_rect(image, bbox, {200, 20, 50});

		//std::cout << textline.dump(4) << std::endl;
	}

	image.write_png("output.png");*/
}

void img_mode(const ImgArgs& args, std::shared_ptr<Channel> channel) {
	/*auto image = stb::Image<stb::RGB>(args.img_file);
	auto js_script = build_js_script_img(args.ref_file);;

	std::cout << "Script: " << js_script << std::endl;

	JSRequest msg(image, js_script);

	channel->send_request(msg);
	auto response = channel->receive_response();

	auto ref_image = stb::Image<stb::RGB>(response.at("data").get<std::string>());
	RefImage ref_msg(ref_image);
	channel->send_request(ref_msg);

	response = channel->receive_response();	
	std::cout << "Response: " << std::endl;
	std::cout << response.dump(4) << std::endl << std::endl;

	auto data = response.at("data");
	for (auto& textline: data) {
		nn::Rect bbox{
			textline.at("left").get<int32_t>(),
			textline.at("top").get<int32_t>(),
			textline.at("right").get<int32_t>(),
			textline.at("bottom").get<int32_t>()
		};
		draw_rect(image, bbox, {200, 20, 50});

		//std::cout << textline.dump(4) << std::endl;
	}

	image.write_png("output.png");*/
}

void js_mode(const JSArgs& args, std::shared_ptr<Channel> channel) {
	/*auto image = stb::Image<stb::RGB>(args.img_file);

	std::ifstream script_file(args.script_file);
	if (!script_file.is_open()) {
		throw std::runtime_error("Failed to open script file");
	}
	std::string script = {
		std::istreambuf_iterator<char>(script_file),
		std::istreambuf_iterator<char>()
	};

	JSRequest msg(image, script);

	channel->send_request(msg);
	auto response = channel->receive_response();

	std::cout << "Response: " << std::endl;
	std::cout << response.dump(4);

	auto type = response.at("type").get<std::string>();
	if (type == "TextTensor") {
		auto tensor = response.get<nn::TextTensor>();
		for (auto& img: tensor.objects) {
			draw_rect(image, img.rect, {200, 20, 50});
		}
		image.write_png("output.png");
	} else if (type == "ImgTensor") {
		auto tensor = response.get<nn::ImgTensor>();
		for (auto& img: tensor.objects) {
			draw_rect(image, img.rect, {200, 20, 50});
		}
		image.write_png("output.png");
	} else if (type == "Point") {
		auto point = response.get<nn::Point>();
		std::cout << "Point: {" << point.x << ", " << point.y << "}\n"; 
	} else if (type == "RefImageRequest") {
		stb::Image<stb::RGB> img;
		try {
			img = stb::Image<stb::RGB>(response.at("path").get<std::string>());
		} catch (const std::exception& error) {}
		RefImage request(img);
		channel->send_request(request);

		response = channel->receive_response();

		auto tensor = response.get<nn::ImgTensor>();
		for (auto& img: tensor.objects) {
			draw_rect(image, img.rect, {200, 20, 50});
		}
		image.write_png("output33.png");
	} else if (type == "Error") {
		std::cout << "Error: " << response.at("message").get<std::string>() << std::endl;
	}*/
}

int handler(const Args& args) {
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
			text_mode(text_args, channel);
		} catch (const std::bad_cast& e) {
			try {
				const ImgArgs& img_args = dynamic_cast<const ImgArgs&>(args);
				img_mode(img_args, channel);
			} catch (const std::bad_cast& e) {
				const JSArgs& js_args = dynamic_cast<const JSArgs&>(args);
				js_mode(js_args, channel);
			}	
		} 
	} catch (const std::exception& error) {
		std::cout << error.what() << std::endl;
	}

	return 0;
}

int main(int argc, char** argv) {
	coro::Application([&]{
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

			JSArgs js_args;
			auto js_spec = (
				command("js").set(selected_mode, mode::js),
				value("search image", js_args.img_file),
				value("script", js_args.script_file),
				required("--nn_service") & value("ip:port of the nn_service", js_args.ip_port)
			);

			auto cli = (text_spec | img_spec | js_spec);

			if (!parse(argc, argv, cli)) {
				std::cout << make_man_page(cli, argv[0]) << std::endl;
				return 1;
			}

			if (selected_mode == mode::text) {
				return handler(text_args);
			} else if (selected_mode == mode::img) {
				return handler(img_args);
			} else if (selected_mode == mode::js) {
				return handler(js_args);
			} else {
				throw;
			} 
		} catch (const std::exception& error) {
			std::cout << error.what() << std::endl;
			return 0;
		}
	}).run();
	
	return 0;
}