
#include "LabelingWu.hpp"
#include <cassert>
#include <climits>

namespace nn {

//Based on "Two Strategies to Speed up Connected Components Algorithms", the SAUF (Scan array union find) variant
//using decision trees
//Kesheng Wu, et al
//Note: rows are encoded as position in the "rows" array to save lookup times
//reference for 4-way: {{-1, 0}, {0, -1}};//b, d neighborhoods
const int G4[2][2] = {{1, 0}, {0, -1}};//b, d neighborhoods
//reference for 8-way: {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}};//a, b, c, d neighborhoods
const int G8[4][2] = {{1, -1}, {1, 0}, {1, 1}, {0, -1}};//a, b, c, d neighborhoods

LabelingWu::LabelingWu(int cols_, int rows_) {
	rows = rows_;
	cols = cols_;
	L.resize(rows * cols);
	I.resize(rows * cols);

	//A quick and dirty upper bound for the maximimum number of labels.  The 4 comes from
	//the fact that a 3x3 block can never have more than 4 unique labels for both 4 & 8-way
	const size_t Plength = 4 * (size_t(rows + 3 - 1)/3) * (size_t(cols + 3 - 1)/3);
	P.resize(Plength);
}

std::vector<Rect> LabelingWu::run(int connectivity) {
	assert(rows == I.rows);
	assert(cols == I.cols);
	assert(rows == L.rows);
	assert(cols == L.cols);
	assert(connectivity == 8 || connectivity == 4);
	P[0] = 0;
	LabelT lunique = 1;
	//scanning phase
	for (int r_i = 0; r_i < rows; ++r_i) {
		LabelT * const Lrow = &L[r_i * cols];
		LabelT * const Lrow_prev = Lrow - cols;
		const PixelT * const Irow = &I[r_i * cols];
		const PixelT * const Irow_prev = Irow - cols;
		LabelT *Lrows[2] = {
			Lrow,
			Lrow_prev
		};
		const PixelT *Irows[2] = {
			Irow,
			Irow_prev
		};
		if (connectivity == 8) {
			const int a = 0;
			const int b = 1;
			const int c = 2;
			const int d = 3;
			const bool T_a_r = (r_i - G8[a][0]) >= 0;
			const bool T_b_r = (r_i - G8[b][0]) >= 0;
			const bool T_c_r = (r_i - G8[c][0]) >= 0;
			for (int c_i = 0; Irows[0] != Irow + cols; ++Irows[0], c_i++) {
				if (!*Irows[0]) {
					Lrow[c_i] = 0;
					continue;
				}
				Irows[1] = Irow_prev + c_i;
				Lrows[0] = Lrow + c_i;
				Lrows[1] = Lrow_prev + c_i;
				const bool T_a = T_a_r && (c_i + G8[a][1]) >= 0   && *(Irows[G8[a][0]] + G8[a][1]);
				const bool T_b = T_b_r                            && *(Irows[G8[b][0]] + G8[b][1]);
				const bool T_c = T_c_r && (c_i + G8[c][1]) < cols && *(Irows[G8[c][0]] + G8[c][1]);
				const bool T_d =          (c_i + G8[d][1]) >= 0   && *(Irows[G8[d][0]] + G8[d][1]);

				//decision tree
				if (T_b) {
					//copy(b)
					*Lrows[0] = *(Lrows[G8[b][0]] + G8[b][1]);
				} else {//not b
					if (T_c) {
						if (T_a) {
							//copy(c, a)
							*Lrows[0] = set_union(*(Lrows[G8[c][0]] + G8[c][1]), *(Lrows[G8[a][0]] + G8[a][1]));
						} else {
							if (T_d) {
								//copy(c, d)
								*Lrows[0] = set_union(*(Lrows[G8[c][0]] + G8[c][1]), *(Lrows[G8[d][0]] + G8[d][1]));
							} else {
								//copy(c)
								*Lrows[0] = *(Lrows[G8[c][0]] + G8[c][1]);
							}
						}
					} else {//not c
						if (T_a) {
							//copy(a)
							*Lrows[0] = *(Lrows[G8[a][0]] + G8[a][1]);
						} else {
							if (T_d) {
								//copy(d)
								*Lrows[0] = *(Lrows[G8[d][0]] + G8[d][1]);
							} else {
								//new label
								*Lrows[0] = lunique;
								P[lunique] = lunique;
								lunique = lunique + 1;
							}
						}
					}
				}
			}
		} else {
			//B & D only
			const int b = 0;
			const int d = 1;
			const bool T_b_r = (r_i - G4[b][0]) >= 0;
			for (int c_i = 0; Irows[0] != Irow + cols; ++Irows[0], c_i++) {
				if (!*Irows[0]) {
					Lrow[c_i] = 0;
					continue;
				}
				Irows[1] = Irow_prev + c_i;
				Lrows[0] = Lrow + c_i;
				Lrows[1] = Lrow_prev + c_i;
				const bool T_b = T_b_r                            && *(Irows[G4[b][0]] + G4[b][1]);
				const bool T_d =          (c_i + G4[d][1]) >= 0   && *(Irows[G4[d][0]] + G4[d][1]);
				if (T_b) {
					if (T_d) {
						//copy(d, b)
						*Lrows[0] = set_union(*(Lrows[G4[d][0]] + G4[d][1]), *(Lrows[G4[b][0]] + G4[b][1]));
					} else {
						//copy(b)
						*Lrows[0] = *(Lrows[G4[b][0]] + G4[b][1]);
					}
				} else {
					if (T_d) {
						//copy(d)
						*Lrows[0] = *(Lrows[G4[d][0]] + G4[d][1]);
					} else {
						//new label
						*Lrows[0] = lunique;
						P[lunique] = lunique;
						lunique = lunique + 1;
					}
				}
			}
		}
	}

	//analysis
	LabelT nLabels = flattenL(lunique);

	std::vector<Rect> rects;
	rects.resize(nLabels - 1);

	for (size_t l = 0; l < rects.size(); ++l) {
		Rect& rect = rects[l];
		rect.left = INT_MAX;
		rect.top = INT_MAX;
		rect.right = INT_MIN;
		rect.bottom = INT_MIN;
	}


	for (int r_i = 0; r_i < rows; ++r_i) {
		LabelT *Lrow_start = &L[r_i * cols];
		LabelT *Lrow_end = Lrow_start + cols;
		LabelT *Lrow = Lrow_start;
		for (int c_i = 0; Lrow != Lrow_end; ++Lrow, ++c_i) {
			const LabelT l = P[*Lrow];
			*Lrow = l;

			if (l) {
				Rect& rect = rects[l - 1];
				rect.left = std::min(rect.left, c_i);
				rect.top = std::min(rect.top, r_i);
				rect.right = std::max(rect.right, c_i);
				rect.bottom = std::max(rect.bottom, r_i);
			}
		}
	}

	return rects;
}

//Find the root of the tree of node i
LabelingWu::LabelT LabelingWu::findRoot(LabelT i) {
	LabelT root = i;
	while (P[root] < root) {
		root = P[root];
	}
	return root;
}

//Make all nodes in the path of node i point to root
void LabelingWu::setRoot(LabelT i, LabelT root) {
	while (P[i] < i) {
		LabelT j = P[i];
		P[i] = root;
		i = j;
	}
	P[i] = root;
}

//Find the root of the tree of the node i and compress the path in the process
LabelingWu::LabelT LabelingWu::find(LabelT i) {
	LabelT root = findRoot(i);
	setRoot(i, root);
	return root;
}

//unite the two trees containing nodes i and j and return the new root
LabelingWu::LabelT LabelingWu::set_union(LabelT i, LabelT j) {
	LabelT root = findRoot(i);
	if (i != j) {
		LabelT rootj = findRoot(j);
		if (root > rootj) {
			root = rootj;
		}
		setRoot(j, root);
	}
	setRoot(i, root);
	return root;
}

//Flatten the Union Find tree and relabel the components
LabelingWu::LabelT LabelingWu::flattenL(LabelT length) {
	LabelT k = 1;
	for (LabelT i = 1; i < length; ++i) {
		if (P[i] < i) {
			P[i] = P[P[i]];
		} else {
			P[i] = k; k = k + 1;
		}
	}
	return k;
}

}
