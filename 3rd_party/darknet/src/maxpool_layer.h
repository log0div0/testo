#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H

#include "image.h"
#include "cuda.h"

typedef layer maxpool_layer;

void resize_maxpool_layer(maxpool_layer *l, int w, int h);
void forward_maxpool_layer(const maxpool_layer l, network net);
void backward_maxpool_layer(const maxpool_layer l, network net);

#ifdef GPU
void forward_maxpool_layer_gpu(maxpool_layer l, network net);
void backward_maxpool_layer_gpu(maxpool_layer l, network net);
#endif

#endif

