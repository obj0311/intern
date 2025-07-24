// https://github.com/zsef123/Connected_components_PyTorch/tree/main
//BSD 3-Clause License
//
//Copyright (c) 2020, the respective contributors, as shown by the AUTHORS file.
//All rights reserved.
//
//Redistribution and use in source and binary forms, with or without
//modification, are permitted provided that the following conditions are met:
//
//1. Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
//2. Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
//3. Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>
#include "device_atomic_functions.h"

namespace {

    template <typename T>
    __device__ __forceinline__ unsigned char hasBit(T bitmap, unsigned char pos) {
        return (bitmap >> pos) & 1;
    }


    __device__ int32_t find(const int32_t* s_buf, int32_t n) {
        while (s_buf[n] != n)
            n = s_buf[n];
        return n;
    }

    __device__ int32_t find_n_compress(int32_t* s_buf, int32_t n) {
        const int32_t id = n;
        while (s_buf[n] != n) {
            n = s_buf[n];
            s_buf[id] = n;
        }
        return n;
    }

    __device__ void union_(int32_t* s_buf, int32_t a, int32_t b)
    {
        bool done;
        do
        {
            a = find(s_buf, a);
            b = find(s_buf, b);

            if (a < b) {
                int32_t old = atomicMin(s_buf + b, a);
                done = (old == b);
                b = old;
            }
            else if (b < a) {
                int32_t old = atomicMin(s_buf + a, b);
                done = (old == a);
                a = old;
            }
            else
                done = true;

        } while (!done);
    }

}

namespace cc_torch
{
    __global__ void init_labeling(int32_t* label, const uint32_t W, const uint32_t H, const uint32_t D);
    __global__ void merge(uint8_t* const img, int32_t* label, uint8_t* last_cube_fg, const uint32_t W, const uint32_t H, const uint32_t D);
    __global__ void compression(int32_t* label, const uint32_t W, const uint32_t H, const uint32_t D);
    __global__ void final_labeling(int32_t* label, uint8_t* last_cube_fg, const uint32_t W, const uint32_t H, const uint32_t D);
}

int torch_cc3d(torch::Tensor& output,const torch::Tensor& input);
