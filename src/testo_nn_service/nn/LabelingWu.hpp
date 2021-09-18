
#pragma once

#include <vector>
#include "Rect.hpp"

namespace nn {

struct LabelingWu {
	using PixelT = uint8_t;
	using LabelT = uint16_t;

	LabelingWu() = default;
	LabelingWu(int cols_, int rows_);
	std::vector<Rect> run(int connectivity = 4);

	std::vector<PixelT> I;
	std::vector<LabelT> L;

private:
	int rows = 0;
	int cols = 0;
	std::vector<LabelT> P;

	LabelT findRoot(LabelT i);
	void setRoot(LabelT i, LabelT root);
	LabelT find(LabelT i);
	LabelT set_union(LabelT i, LabelT j);
	LabelT flattenL(LabelT length);
};

}
