#pragma once

#include <torch/script.h>
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "MartianAIVar.h"

using namespace torch::indexing;
using namespace std;

class CommonProcess
{
public:
	CommonProcess(int isCPUOnly, std::string deviceString);
	~CommonProcess();

	int gaussian3DFilter(at::Tensor& input, at::Tensor& output, int kernelSize, int GPUAcc);        // 1 x 1 x 3 x 3 x 3  Tensor
	int dilation3D(at::Tensor& input, at::Tensor& output, int labelNum, int volumeDimA, int volumeDimB, int volumeDimC, int* processOnLabel, int* labelPriorityTable, int GPUAcc);
	int erosion3D(at::Tensor& input, at::Tensor& output, int labelNum, int volumeDimA, int volumeDimB, int volumeDimC, int* processOnLabel, int GPUAcc);
	int torchPCA(at::Tensor& input, int TargetLabel, at::Tensor& eVec, at::Tensor& sampleMean, at::Tensor& eValue, int volumeDimA, int volumeDimB, int volumeDimC);
	int toSingleLabels(at::Tensor& input, at::Tensor& output, int labelNum, int* labelIndex, int volumeDimA, int volumeDimB, int volumeDimC);
	int printTensorInfo(at::Tensor t);

private:
	int mIsCPUOnly = 1;
	std::string mDeviceString = "cpu";

private:

};

