#include <nlohmann/json.hpp>
#include <stb/Image.hpp>

#include <string>

struct Request {
	Request() = default;
	virtual ~Request() = default;
	Request(const stb::Image<stb::RGB>& screenshot): screenshot(screenshot) {
		header["version"] = NN_SERVICE_PROCOTOL_VERSION;
		update_header("screenshot", screenshot);
	}

	nlohmann::json header;
	stb::Image<stb::RGB> screenshot;
protected:
	void update_header(const std::string& field, const stb::Image<stb::RGB>& pic);
};

void Request::update_header(const std::string& field, const stb::Image<stb::RGB>& pic) {
	header[field] = nlohmann::json::object();

	header[field]["w"] = pic.w;
	header[field]["h"] = pic.h;
	header[field]["c"] = pic.c;
}


struct TextRequest: Request {
	TextRequest() = default;
	TextRequest(const stb::Image<stb::RGB>& screenshot,
			std::string text_to_find = "",
			std::string color_fg = "",
			std::string color_bg = ""): Request(screenshot)
	{
		header["type"] = "text";
		if (text_to_find.length()) {
			header["text_to_find"] = text_to_find;
		}

		if (color_fg.length()) {
			header["color_fg"] = color_fg;
		}

		if (color_bg.length()) {
			header["color_bg"] = color_bg;
		}
	}
	
	bool has_text() const {
		return header.count("text_to_find");
	}

	std::string text() const {
		return header.at("text_to_find").get<std::string>();
	}

};

struct ImgRequest: Request {
	ImgRequest() = default;
	ImgRequest(const stb::Image<stb::RGB>& screenshot, const stb::Image<stb::RGB>& pattern): Request(screenshot), pattern(pattern)
	{
		header["type"] = "img";
		update_header("pattern", pattern);
	}

	stb::Image<stb::RGB> pattern;
};
