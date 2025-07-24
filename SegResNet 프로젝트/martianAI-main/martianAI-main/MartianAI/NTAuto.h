#pragma once

#include <torch/script.h>
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "MartianAIVar.h"
#include "ControlSharedMemory.h"
#include "CommonProcess.h"

using namespace torch::indexing;
using namespace std;

#define NT_ERR_SUCCESS       400
#define NT_OUTPUT_NOT_EXIST  401
#define NT_OUTPUT_TOO_SMALL  402

enum NT_IDX { NT_IDX_HEAD = 0, NT_IDX_BG = 0, NT_IDX_MID, NT_IDX_NT, NT_IDX_DCP, NT_IDX_NB };
enum NT_NUM { NT_NUM_BG = 0, NT_NUM_HEAD = 2, NT_NUM_MID = 3, NT_NUM_NT = 5, NT_NUM_DCP, NT_NUM_NB };

class NTAuto
{
public:
	NTAuto(int isCPUOnly, std::string deviceString, CommonProcess* commonProcess);
	~NTAuto();
	int ExcuteNTAuto(torch::jit::script::Module module_head, torch::jit::script::Module module_inside, at::Tensor& input, InferenceInfo* inferenceInfo, unsigned char* outputBuffer);
	int Measure2DSagmentation(torch::jit::script::Module module, at::Tensor& input, at::Tensor& output, float* pcaVector, int width, int height, bool RGB);

private:
	int mIsCPUOnly = 1;
	std::string mDeviceString = "cpu";
	CommonProcess* mCommonProcess;
	const int labelNum_NT_head = 1;
	const int labelNum_NT_inside = 5;
	char labelName_NT[NUM_LABEL_NT][4] = { "HD", "BG",  "ML", "NT", "DCP", "NB" };
	int labelOffsetTable_NT[NUM_LABEL_NT] = { 2, 0, 3, 5, 6, 7 }; // itk-snap visualize color (hd, bg, mid, nt, dcp, nb)
	float labelThreshold_NT[NUM_LABEL_NT] = { 0.5,0.5,0.5,0.5,0.5,0.5 }; // segmentation output postprocessing -> tresholding
	int processOnLabel_NT[NUM_LABEL_NT] = { 0, 0, 0, 0, 0, 0 }; // segmentation output postprocessing -> dilation(1)/ erosion(2)/ both(3)
	int labelPriorityTable_NT[NUM_LABEL_NT] = { 1, 0, 5, 3, 4, 2 }; // morpology operation label priority
	
	int ccLabelMap_NT[NUM_LABEL_NT] = { 20, 0, 40, 60, 80, 100 }; // ccl seperate label
	int labelMap_NT[NUM_LABEL_NT] = { 1, 2, 3, 4, 5, 6 }; // error label return (hd, bg, mid, nt, dcp, nb)
	int processOrderInside[NUM_LABEL_NT - 1] = { 0, 1, 3, 2, 4}; // post-process order bg(0), mid(1) -> dcp(3) -> nt(2) -> nb(4) 
	bool RuleLabelExist_NT[NUM_LABEL_NT] = { true, false, true, false, false, false }; // post-process rule baed existing label
	int RuleMaxComponentCount_NT[NUM_LABEL_NT] = { 1, 0, 1, 1, 1, 1 }; // post-process rule baed maximum number of component for label
	unsigned int RuleMinVolumeSize_NT[NUM_LABEL_NT] = { 3000, 0, 400, 0, 0, 0 }; //post-process rule baed minimum size of label
	at::Tensor CcTensor;
	int HeadCenter[3] = {0,};
	struct ComponentInfo { UINT32 volume; UINT32 ccLabel; float centroid[3]; };
	//vector<ComponentInfo> ntLabelsInfo[NUM_LABEL_NT];

private:
	int segResnet_head(torch::jit::script::Module module, at::Tensor& input, unsigned char* outputBuffer, float* SegMeanOut, int volumeDimA, int volumeDimB, int volumeDimC, int gauissianFilterMode, int morphProcessMode, int* errorCode, int* returnValue);
	int segResnet_inside(torch::jit::script::Module module, at::Tensor& input, at::Tensor& output, unsigned char* outputBuffe, float* SegMeanOut, int volumeDimA, int volumeDimB, int volumeDimC, int gauissianFilterMode, int morphProcessMode, int* errorCode, int* returnValue);
	int cc3d(at::Tensor& outTensor, at::Tensor& inTensor, vector<ComponentInfo>& labelInfo, UINT32 mappedLabel, bool returnCentroid);
	int removeComponent(at::Tensor& outTensor, at::Tensor& inTensor, int labelIndex, int targetLabel);
	int removeExcessComponents(at::Tensor& outTensor, at::Tensor& inTensor, int labelIndex, vector<ComponentInfo>& labelInfo, int startIndex, int endIndex);
	int HeadMaximumRadius(at::Tensor inputData, float* MaximumRadius);
};