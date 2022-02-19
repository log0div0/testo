
#pragma once

#include "ReportWriterNative.hpp"
#include <coro/StreamSocket.h>

struct ReportWriterNativeRemote: ReportWriterNative {
	ReportWriterNativeRemote(const ReportConfig& config);

	virtual void launch_begin(const std::vector<std::shared_ptr<IR::Test>>& tests,
		const std::vector<std::shared_ptr<IR::TestRun>>& tests_runs) override;

	virtual void test_skip_begin(const std::shared_ptr<IR::TestRun>& test_run) override;
	virtual void test_skip_end(const std::shared_ptr<IR::TestRun>& test_run) override;

	virtual void test_begin(const std::shared_ptr<IR::TestRun>& test_run) override;
	virtual void report(const std::shared_ptr<IR::TestRun>& test_run, const std::string& text) override;
	virtual void report_screenshot(const std::shared_ptr<IR::TestRun>& test_run, const stb::Image<stb::RGB>& screenshot) override;
	virtual void test_end(const std::shared_ptr<IR::TestRun>& test_run) override;

	virtual void launch_end() override;

private:
	using Socket = coro::StreamSocket<asio::ip::tcp>;
	using Endpoint = asio::ip::tcp::endpoint;

	Socket socket;
	Endpoint endpoint;

	nlohmann::json recv();
	void send(const nlohmann::json& message);
	void wait_for_confirmation();
};
