
#include "Homm3Detector.hpp"
#include <iostream>
#include <algorithm>
#include <list>

namespace nn {

static inline bool is_n_times_div_by_2(int value, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		if ((value % 2) != 0) {
			return false;
		}
		value /= 2;
	}
	return true;
}

static inline int nearest_n_times_div_by_2(int value, size_t n) {
	while (true) {
		if (is_n_times_div_by_2(value, n)) {
			return value;
		}
		value += 1;
	}
}

Homm3Detector& Homm3Detector::instance() {
	static Homm3Detector instance;
	return instance;
}

std::vector<Homm3Object> Homm3Detector::detect(const stb::Image<stb::RGB>* image) {
	if (!image->data) {
		return {};
	}

	run_nn(image);
	return run_postprocessing();
}

void Homm3Detector::run_nn(const stb::Image<stb::RGB>* image) {
	if ((in_w != image->w) ||
		(in_h != image->h))
	{
		in_h = image->h;
		in_w = image->w;
		int in_pad_h = nearest_n_times_div_by_2(in_h, 5);
		int in_pad_w = nearest_n_times_div_by_2(in_w, 5);

		pred_count =
			((in_pad_w >> 5) * (in_pad_h >> 5)) * 3 +
			((in_pad_w >> 4) * (in_pad_h >> 4)) * 3;
		pred_length = 5 + Homm3Object::classes_names.size();

		in.resize(in_pad_w, in_pad_h, 3);
		out.resize(pred_count, pred_length);
		in.fill(0);
	}

	in.set(*image, true);

	model.run({&in}, {&out});
}

struct Homm3Prediction {
	Homm3Prediction(float* data) {
		float x = data[0];
		float y = data[1];
		float w = data[2];
		float h = data[3];
		x1 = x - w / 2;
		y1 = y - h / 2;
		x2 = x + w / 2;
		y2 = y + h / 2;
		conf = data[4];
		auto it = std::max_element(data + 5, data + 5 + Homm3Object::classes_names.size());
		class_index = std::distance(data + 5, it);
	}

	float iou(const Homm3Prediction& other) const {
		float inter_rect_x1 = std::max(x1, other.x1);
		float inter_rect_y1 = std::max(y1, other.y1);
		float inter_rect_x2 = std::min(x2, other.x2);
		float inter_rect_y2 = std::min(y2, other.y2);
		float inter_area =
			std::max(inter_rect_x2 - inter_rect_x1 + 1, 0.0f) *
			std::max(inter_rect_y2 - inter_rect_y1 + 1, 0.0f);

		float area = (x2 - x1 + 1) * (y2 - y1 + 1);
		float other_area = (other.x2 - other.x1 + 1) * (other.y2 - other.y1 + 1);

		float iou = inter_area / (area + other_area - inter_area + 1e-16);
		return iou;
	}

	float x1 = 0, y1 = 0, x2 = 0, y2 = 0, conf = 0;
	int class_index = 0;
};

#define CONF_THRES 0.7f
#define NMS_THRES 0.4f

std::vector<Homm3Object> Homm3Detector::run_postprocessing() {
	Rect img_rect;
	img_rect.left = 0;
	img_rect.top = 0;
	img_rect.right = in_w - 1;
	img_rect.bottom = in_h - 1;
	std::vector<Homm3Object> result;
	std::list<Homm3Prediction> preds;
	for (int i = 0; i < pred_count; ++i) {
		if (out[i][4] < CONF_THRES) {
			continue;
		}
		preds.push_back(Homm3Prediction(out[i]));
	}
	preds.sort([](const Homm3Prediction& a, const Homm3Prediction& b) {
		return a.conf > b.conf;
	});
	while (preds.size()) {
		Homm3Prediction pred = preds.front();
		float x1 = pred.x1 * pred.conf;
		float y1 = pred.y1 * pred.conf;
		float x2 = pred.x2 * pred.conf;
		float y2 = pred.y2 * pred.conf;
		float weights = pred.conf;
		preds.pop_front();
		for (auto it = preds.begin(); it != preds.end();) {
			if (it->class_index != pred.class_index) {
				++it;
				continue;
			}
			if (it->iou(pred) < NMS_THRES) {
				++it;
				continue;
			}
			x1 += it->x1 * it->conf;
			y1 += it->y1 * it->conf;
			x2 += it->x2 * it->conf;
			y2 += it->y2 * it->conf;
			weights += it->conf;
			preds.erase(it++);
		}
		Homm3Object obj;
		obj.rect.left = x1 / weights;
		obj.rect.top = y1 / weights;
		obj.rect.right = x2 / weights;
		obj.rect.bottom = y2 / weights;
		obj.rect = obj.rect & img_rect;
		obj.class_name = Homm3Object::classes_names.at(pred.class_index);
		result.push_back(obj);
	}
	return result;
}

}
