
#pragma once

#include <vbox/virtual_box.hpp>
#include <memory>

#ifdef WIN32
namespace darknet {
struct API {
	bool match(const uint8_t* data, size_t data_size, const std::string& text) const {
		throw std::runtime_error("darknet api is not implemented yet");
	}
};
}
#else
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <libvirt/libvirt.h>
#include <iostream>
namespace darknet {
struct API {
	API(): tesseract_api(new tesseract::TessBaseAPI()) {
		if (tesseract_api->Init(nullptr, "eng")) {
			throw std::runtime_error("Tess api init fail");
		}
	}
	~API() {
		if (tesseract_api) {
			tesseract_api->End();
		}
	}
	bool match(const uint8_t* data, size_t data_size, const std::string& text) const {
		PIX* img = pixReadMemPng(data, data_size);
		tesseract_api->SetImage(img);

		auto out_text = tesseract_api->GetUTF8Text();
		std::string out_string(out_text);
		delete [] out_text;
		pixDestroy(&img);

		std::cout << out_string << std::endl;

		return out_string.find(text) != std::string::npos;
	}

private:
	std::unique_ptr<tesseract::TessBaseAPI> tesseract_api;
};
}
#endif

struct API {
public:
	static API& instance() {
		static API APIInstance;
		return APIInstance;
	}

	API() = default;
	~API() = default;

	darknet::API darknet_api;
	vbox::API vbox_api;
private:
	API(const API&) = delete;
	API& operator=(const API&) = delete;
};