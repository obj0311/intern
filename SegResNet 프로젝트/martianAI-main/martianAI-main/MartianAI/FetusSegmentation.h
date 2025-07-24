#pragma once

#include <torch/script.h>
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "MartianAIVar.h"
#include "ControlSharedMemory.h"
#include "CommonProcess.h"

using namespace torch::indexing;
using namespace std;

class FetusSegmentation
{
public:
	FetusSegmentation(int isCPUOnly, std::string deviceString, CommonProcess* commonProcess);
	~FetusSegmentation();

	int ExcuteFetusSegmentation(torch::jit::script::Module module, at::Tensor& input, int mode, InferenceInfo* inferenceInfo, unsigned char* output);

private:
    int mIsCPUOnly = 1;
    std::string mDeviceString = "cpu";
    CommonProcess* mCommonProcess;

    // 1st trimester
    // fluid      - 90
    // fetus      - 18
    // ucord      - 36
    // uterus     - 72
    // placenta   - 54

    int labelOffsetTable_1st[6] = { 0, 90, 18, 36, 72, 54 };
    float labelThreshold_1st[6] = { 0.5,0.5,0.5,0.5,0.5,0.5 };
    int labelPriorityTable_1st[6] = { 0, 3, 14, 12, 5, 7 };
    int processOnLabel_1st[6] = { 0,0,0,0,0,0 };
    int interpolationOrder_1st[6] = { 0, 1, 4, 5, 3, 2 };
    char mLabelName1st[6][4] = { "BG", "FL", "FE", "UC", "UT", "PL" };

    // 2nd trimester
    // fluid      - 234
    // face       - 108
    // body       - 180
    // ucord      - 126
    // limbs      - 144
    // uterus     - 216
    // placenta   - 198
    int labelOffsetTable_2nd[8] = { 0, 234, 108, 180, 126, 144, 216, 198 };
    int labelPriorityTable_2nd[8] = { 0, 4,  15,   9,  13,  11,   6,   8 };
    float labelThreshold_2nd[8] = { 0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5 };
    int processOnLabel_2nd[8] = { 0,0,0,0,0,0,0,0 };
    int interpolationOrder_2nd[8] = { 0, 7, 1, 8, 9, 3, 4, 2 };
    char mLabelName2nd[8][4] = { "BG", "FL", "FA", "BO", "UC", "LI", "UT", "PL" };

private:
    int FetusSegmentation::segResnet(torch::jit::script::Module module, at::Tensor& input, at::Tensor& output, unsigned char* outputBuffer, float* SegMeanOut, int mode, int labelNum, float* labelThreshold, int* processOnLabel, int* labelPriorityTable, int* labelOffsetTable, int volumeDimA, int volumeDimB, int volumeDimC, int gauissianFilterMode, int morphProcessMode, int* errorCode, int* returnValue);
    int FetusSegmentation::countVoxels(at::Tensor& input, float* SegMeanOut, int labelNum, int* labelOffsetTable, char labelName[][4]);
};

