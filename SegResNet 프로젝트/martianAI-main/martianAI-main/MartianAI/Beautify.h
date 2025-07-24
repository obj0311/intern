#pragma once
#include <torch/script.h>
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "MartianAIVar.h"

using namespace torch::indexing;
using namespace std;

class Beautify
{
public:
	Beautify(int isCPUOnly, std::string device_string);
	~Beautify();

	int StyleTransfer(torch::jit::script::Module module, at::Tensor& input, at::Tensor& output, int width, int height, double rotate);
	int FaceClassification(torch::jit::script::Module module, at::Tensor& input, at::Tensor& output, int width, int height, int* FaceProb);

private:
	int mIsCPUOnly = 1;
	std::string mDeviceString = "cpu";
};

