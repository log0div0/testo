#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}

void forward_convolutional_layer(convolutional_layer l, network net)
{
    int i;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        float *a = l.weights;
        float *b = net.workspace;
        float *c = l.output + i*n*m;
        float *im =  net.input + i*l.c*l.h*l.w;

        if (l.size == 1) {
            b = im;
        } else {
            im2col_cpu(im, l.c, l.h, l.w, l.size, l.stride, l.pad, b);
        }
        gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
    }

    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
    }

    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_convolutional_layer(convolutional_layer l, network net)
{
    int i;
    int m = l.n;
    int n = l.size*l.size*l.c;
    int k = l.out_w*l.out_h;

    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }

    for(i = 0; i < l.batch; ++i){
        float *a = l.delta + i*m*k;
        float *b = net.workspace;
        float *c = l.weight_updates;

        float *im  = net.input + i*l.c*l.h*l.w;
        float *imd = net.delta + i*l.c*l.h*l.w;

        if(l.size == 1){
            b = im;
        } else {
            im2col_cpu(im, l.c, l.h, l.w,
                    l.size, l.stride, l.pad, b);
        }

        gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

        if (net.delta) {
            a = l.weights;
            b = l.delta + i*m*k;
            c = net.workspace;
            if (l.size == 1) {
                c = imd;
            }

            gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

            if (l.size != 1) {
                col2im_cpu(net.workspace, l.c, l.h, l.w, l.size, l.stride, l.pad, imd);
            }
        }
    }
}

void update_convolutional_layer(convolutional_layer l, network net)
{
    float learning_rate = net.learning_rate;
    float momentum = net.momentum;
    float decay = net.decay;
    int batch = net.batch;

    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.nweights, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.nweights, momentum, l.weight_updates, 1);
}
