#pragma once

#include <torch/script.h>
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "MartianAIVar.h"
#include "ControlSharedMemory.h"
#include "CommonProcess.h"

using namespace torch::indexing;
using namespace std;

enum CNS_IDX { CNS_IDX_BG = 0, CNS_IDX_TH, CNS_IDX_CB, CNS_IDX_CM, CNS_IDX_CP, CNS_IDX_PVC, CNS_IDX_CSP, CNS_IDX_MID };
enum CNS_NUM { CNS_NUM_BG = 1, CNS_NUM_TH, CNS_NUM_CB, CNS_NUM_CM, CNS_NUM_CP, CNS_NUM_PVC, CNS_NUM_CSP, CNS_NUM_MID };

class CNSAuto
{
public:
	CNSAuto(int isCPUOnly, std::string deviceString, CommonProcess* commonProcess);
	~CNSAuto();

	int ExcuteCNSAuto(torch::jit::script::Module module, at::Tensor& input, InferenceInfo* inferenceInfo, unsigned char* output);
    int mmSeg(torch::jit::script::Module module, at::Tensor& input, at::Tensor& output, int width, int height, int channel);

private:
	int mIsCPUOnly = 1;
	std::string mDeviceString = "cpu";
    CommonProcess* mCommonProcess;

    // member variable for CNS
    char labelName_CNS[NUM_LABEL_CNS][4] = { "BG", "Th", "CB", "CM", "CP", "PVC", "CSP", "Mid" };
    // The number of rules is based on 145 zoom and 128 volume size.
    bool RuleLabelExist_CNS[NUM_LABEL_CNS] = { false, true, true, false, true, true, true, true };
    int RuleMaxComponentCount_CNS[NUM_LABEL_CNS] = { 0,2,2,3,1,1,1,5 };
    //int RuleVolumeSize_CNS[NUM_LABEL_CNS][2] = { {0,0},/*Th*/{120,2050},/*CB*/{80,7000},/*CM*/{200,3000},/*CP*/{500,3000},/*PVC*/{120,2800},/*CSP*/{120,1700},/*Mid*/{100,5000} }; // { {min, max}, ...}
    int RuleVolumeSize_CNS[NUM_LABEL_CNS][2] = { {0,0},/*Th*/{80,2050},/*CB*/{60,7000},/*CM*/{20,3000},/*CP*/{200,3000},/*PVC*/{80,2800},/*CSP*/{80,1700},/*Mid*/{60,5000} }; // { {min, max}, ...}
    int RuleDistance_CNS[NUM_LABEL_CNS][3] = { {-1,-1,-1}, {-1,-1,-1}, {1,12,31}, {-1,-1,-1}, {-1,-1,-1}, {4,6,40}, {1,13,33}, {-1,-1,-1} }; // { {associated label index, min, max}, ...}
    int RuleDistanceFromMidPlane_CNS[NUM_LABEL_CNS] = { -1,-1,-1,-1,-1,-1,-1,10 };
    int RuleMinVolumeSize[NUM_LABEL_CNS] = { -1, -1, -1, -1, -1, -1, 50, -1 };
    int processOrder_CNS[NUM_LABEL_CNS] = { 0, 1, 6, 2, 3, 4, 5, 7 }; // process order Th->CSP->CB->CM->CP->PVC->Mid
    int ccLabelMap_CNS[NUM_LABEL_CNS] = { 0, 20, 40, 60, 80, 100, 120, 140 };
    int labelMap_CNS[NUM_LABEL_CNS] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    int labelOffsetTable_CNS[NUM_LABEL_CNS] = { 0, 2, 3, 4, 5, 6, 7, 8 };
    int labelPriorityTable_CNS[NUM_LABEL_CNS] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    float labelThreshold_CNS[NUM_LABEL_CNS] = { 0.0,0.5,0.5,0.5,0.5,0.5,0.5,0.5 };
    int processOnLabel_CNS[NUM_LABEL_CNS] = { 0,0,0,0,0,0,0,0 };
    at::Tensor CcTensor;
    struct ComponentInfo { UINT32 volume; UINT32 ccLabel; float centroid[3]; };
    vector<ComponentInfo> cnsLabelsInfo[NUM_LABEL_CNS];
    float SymmetryThreshold_CNS = 0.4;
    boolean hasCNSPlanes = false;
    at::Tensor plane_n[3];
    at::Tensor plane_d[3];
    vector<array<float, 3>> thK2;
    vector<array<float, 3>> cbK2;
    at::Tensor mid_plane_n;
    at::Tensor mid_plane_d;

private:
    int segResnet(torch::jit::script::Module module, at::Tensor& input, at::Tensor& output, unsigned char* outputBuffer, float* SegMeanOut, int mode, int labelNum, float* labelThreshold, int* processOnLabel, int* labelPriorityTable, int* labelOffsetTable, int volumeDimA, int volumeDimB, int volumeDimC, int gauissianFilterMode, int morphProcessMode, int* errorCode, int* returnValue);

    int torchPCAProjectionAnalysis(at::Tensor& input, int TargetLabel, float* axisRadius, int volumeDimA, int volumeDimB, int volumeDimC);
    int torchPCAMeans(at::Tensor& input, int TargetLabel, int axis, float* means, int volumeDimA, int volumeDimB, int volumeDimC);
    int torchKMeans(at::Tensor& input, at::Tensor& CC, int clusterNum);
    int cc3d(at::Tensor& outTensor, at::Tensor& inTensor, vector<ComponentInfo>& labelInfo, UINT32 mappedLabel, bool returnCentroid);
    int removeComponent(at::Tensor& outTensor, at::Tensor& inTensor, int labelIndex, int targetLabel);
    int removeOutlier(at::Tensor& outTensor, at::Tensor& inTensor, int labelIndex, vector<ComponentInfo>& labelInfo, UINT32 threshold);
    int removeExcessComponents(at::Tensor& outTensor, at::Tensor& inTensor, int labelIndex, vector<ComponentInfo>& labelInfo, int startIndex, int endIndex);
    int removeComponentFarFromMidPlane(at::Tensor& outTensor, at::Tensor& inTensor, vector<ComponentInfo>* labelsInfo, int targetLabelIndex, int limit);
    int checkInRange(int labelIndex, vector<ComponentInfo>* labelsInfo, int* errorCode, int* returnValue);
    int setPlanes(IN vector<ComponentInfo>* labelsInfo, IN vector<array<float, 3>> thK2);
    int checkDistanceFromMidPlane(IN vector<vector<ComponentInfo>> labelsInfo, IN int targetLabelIndex, IN int limit, int* errorCode, int* returnValue);
    int checkSymmetry(at::Tensor n, at::Tensor d, at::Tensor p1, at::Tensor p2, IN int targetLabelIndex, int* errorCode, int* returnValue);
};

