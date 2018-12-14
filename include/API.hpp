
#pragma once

#include <tesseract/baseapi.h>
#include <vbox/virtual_box.hpp>
#include <memory>

struct API {
public:
	static API& instance() {
		static API APIInstance;
		return APIInstance;
	}

	~API() {
		if (tesseract_api) {
			tesseract_api->End();
		}
	}

	std::unique_ptr<tesseract::TessBaseAPI> tesseract_api;
	vbox::API vbox_api;

private:
	API(): tesseract_api(new tesseract::TessBaseAPI()) {
		if (tesseract_api->Init(nullptr, "eng")) {
			throw std::runtime_error("Tess api init fail");
		}
	}

	API(const API&) = delete;
	API& operator=(const API&) = delete;
};