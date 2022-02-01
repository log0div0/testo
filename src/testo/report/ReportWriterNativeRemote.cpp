
#include "ReportWriterNativeRemote.hpp"

asio::ip::tcp::endpoint parse_tcp_endpoint(const std::string& endpoint);

ReportWriterNativeRemote::ReportWriterNativeRemote(const ReportConfig& config): ReportWriterNative(config) {
	endpoint = parse_tcp_endpoint(config.report_folder);
}

void ReportWriterNativeRemote::launch_begin(const std::vector<std::shared_ptr<IR::Test>>& tests,
	const std::vector<std::shared_ptr<IR::TestRun>>& tests_runs)
{
	ReportWriterNative::launch_begin(tests, tests_runs);

	nlohmann::json tests_meta = nlohmann::json::array();
	for (auto& test: tests) {
		tests_meta.push_back(to_json(test));
	}

	socket.connect(endpoint);
	send({
		{"type", "launch_begin"},
		{"current_launch", current_launch_meta},
		{"tests", tests_meta},
	});
	wait_for_confirmation();
}

void ReportWriterNativeRemote::test_skip_begin(const std::shared_ptr<IR::TestRun>& test_run) {
	ReportWriterNative::test_skip_begin(test_run);
	send({
		{"type", "test_skip_begin"},
		{"current_test_run", to_json(test_run)},
	});
	wait_for_confirmation();
}

void ReportWriterNativeRemote::test_skip_end(const std::shared_ptr<IR::TestRun>& test_run) {
	ReportWriterNative::test_skip_end(test_run);
	send({
		{"type", "test_skip_end"},
		{"current_test_run", to_json(test_run)},
	});
	wait_for_confirmation();
}

void ReportWriterNativeRemote::test_begin(const std::shared_ptr<IR::TestRun>& test_run) {
	ReportWriterNative::test_begin(test_run);
	send({
		{"type", "test_begin"},
		{"current_test_run", to_json(test_run)},
	});
	wait_for_confirmation();
}

void ReportWriterNativeRemote::report(const std::shared_ptr<IR::TestRun>& test_run, const std::string& text) {
	nlohmann::json msg = {
		{"type", "report"},
		{"text", text},
	};
	if (test_run) {
		msg["current_test_run"] = to_json(test_run);
	}
	send(msg);
	wait_for_confirmation();
}

void ReportWriterNativeRemote::report_screenshot(const std::shared_ptr<IR::TestRun>& test_run, const stb::Image<stb::RGB>& screenshot, const std::string& tag) {
	nlohmann::json msg = {
		{"type", "report_screenshot"},
		{"screenshot", screenshot.write_png_mem()},
		{"tag", tag},
	};
	if (test_run) {
		msg["current_test_run"] = to_json(test_run);
	}
	send(msg);
	wait_for_confirmation();
}

void ReportWriterNativeRemote::test_end(const std::shared_ptr<IR::TestRun>& test_run) {
	ReportWriterNative::test_end(test_run);
	send({
		{"type", "test_end"},
		{"current_test_run", to_json(test_run)},
	});
	wait_for_confirmation();
}

void ReportWriterNativeRemote::launch_end() {
	ReportWriterNative::launch_end();
	send({
		{"type", "launch_end"},
		{"current_launch", current_launch_meta},
	});
	wait_for_confirmation();
}

nlohmann::json ReportWriterNativeRemote::recv() {
	uint32_t msg_size;

	socket.read((uint8_t*)&msg_size, 4);

	std::vector<uint8_t> json_data;
	json_data.resize(msg_size);
	socket.read((uint8_t*)json_data.data(), json_data.size());

	return nlohmann::json::from_cbor(json_data);
}

void ReportWriterNativeRemote::send(const nlohmann::json& json) {
	std::vector<uint8_t> json_data = nlohmann::json::to_cbor(json);
	uint32_t json_size = (uint32_t)json_data.size();
	socket.write((uint8_t*)&json_size, sizeof(json_size));
	socket.write((uint8_t*)json_data.data(), json_size);
}

void ReportWriterNativeRemote::wait_for_confirmation() {
	nlohmann::json response = recv();
	if (response.at("type") != "confirmation") {
		throw std::runtime_error("Got unexpected response from the tcp report server");
	}
}
