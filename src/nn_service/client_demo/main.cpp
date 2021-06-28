#include <iostream>
#include <stdexcept>
#include <thread>
#include <chrono>

#include <coro/Application.h>
#include <coro/StreamSocket.h>

#include "../Channel.hpp"
#include "../../nn/TextTensor.hpp"

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

void local_handler() {
	try {
		auto endpoint = Endpoint(asio::ip::address::from_string("127.0.0.1"), 8888);
		Socket socket;
		socket.connect(endpoint);
		std::cout << "Connected\n";

		std::shared_ptr<Channel> channel(new TCPChannel(std::move(socket)));

		auto image = stb::Image<stb::RGB>("temp2.png");
		TextRequest msg(image, "definition");
		channel->send_request(msg);
		auto response = channel->receive_response();
		std::cout << "Response: " << std::endl;
		std::cout << response.dump(4);
		auto tensor = response.get<nn::TextTensor>();

		for (auto& textline: tensor.objects) {
			draw_rect(image, textline.rect, {200, 20, 50});
		}

		image.write_png("outputlala.png");
	} catch (const std::exception& error) {
		std::cout << error.what() << std::endl;
	}
	

	/*acceptor.run([](coro::StreamSocket<asio::ip::tcp> socket) {
		try {
			std::cout << "Accepted\n";

			std::shared_ptr<Channel> channel(new TCPChannel(std::move(socket)));
			while (true) {
				auto screenshot = channel->receive();
				std::cout << "Received screenshot\n";
			}
			//MessageHandler message_handler(std::move(channel));
			//message_handler.run();
		} catch (const std::exception& error) {
			std::cout << "Error inside local acceptor loop: " << error.what();
		}
	});*/
}



int main(int argc, char** argv) {
	try {
		std::thread t1([]() {coro::Application(local_handler).run();});
		t1.join();
	} catch (const std::exception& error) {
		std::cout << error.what() << std::endl;
	}
	return 0;
}