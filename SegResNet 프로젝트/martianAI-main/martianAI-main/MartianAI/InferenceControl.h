#pragma once
#define NOMINMAX

#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>
#include <vector>
#include <windows.h>
#include "ControlSharedMemory.h"
#include "PelvicAssist.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include "MartianAIVar.h"
#include "Encryption.h"
#include "Beautify.h"
#include "CNSAuto.h"
#include "FetusSegmentation.h"
#include "CommonProcess.h"
#include "NTAuto.h"

using namespace torch::indexing;
using namespace std;

class InferenceControl
{
public:

    InferenceControl(void);
    ~InferenceControl();
    int RunInference(int Size, int Mode);
    int InitInference(int device, int gpuSelection, int ModelPathLen, WCHAR* ModelPath);

    bool InitFlag = 0;
    std::string mDeviceString;
    int isCPUOnly = 0;

private:
    Beautify* mBeautify;
    CNSAuto* mCNSAuto;
    FetusSegmentation* mFetusSegmentation;
    PelvicAssist* mPelvicAssist;
    CommonProcess* mCommonProcess;
    NTAuto* mNTAuto;

    unsigned char* m_InputBuffer = NULL;
    unsigned char* m_OutputBuffer = NULL;

    ControlSharedMemory* m_ControlSharedMemory = NULL;

    std::wstring m_ModelFileFetusSeg1st;
    std::wstring m_ModelFileFetusSeg2nd;
    std::wstring m_ModelFileCNSSeg;
    std::wstring m_ModelFileSTInpainting;
    std::wstring m_ModelFileSTEnhancement;
    std::wstring m_ModelFileSTAutoRestore;
    std::wstring m_ModelFileSTFaceClassification;
    std::wstring m_ModelFileCNSMTC;
    std::wstring m_ModelFileCNSMTT;
    std::wstring m_ModelFileCNSMTV;
    std::wstring m_ModelFileNTSeg_head;
	std::wstring m_ModelFileNTSeg_inside;
	std::wstring m_ModelFileNTMeasure;
    std::wstring m_ModelFilePelvicAssistSeg;
    std::wstring m_ModelFilePelvicAssistMeasure1;
    std::wstring m_ModelFilePelvicAssistMeasure2;

    torch::jit::script::Module m_ModuleFetusSeg1st;
    torch::jit::script::Module m_ModuleFetusSeg2nd;
    torch::jit::script::Module m_ModuleCNSSeg;
    torch::jit::script::Module m_ModuleSTInpainting;
    torch::jit::script::Module m_ModuleSTEnhancement;
    torch::jit::script::Module m_ModuleSTAutoRestore;
    torch::jit::script::Module m_ModuleSTFaceClassification;
    torch::jit::script::Module m_ModuleCNSMTC;
    torch::jit::script::Module m_ModuleCNSMTT;
    torch::jit::script::Module m_ModuleCNSMTV;
    torch::jit::script::Module m_ModuleNTSeg_head;
    torch::jit::script::Module m_ModuleNTSeg_inside;
    torch::jit::script::Module m_ModuleNTMeasure;
    torch::jit::script::Module m_ModulePelvicAssistSeg;
    torch::jit::script::Module m_ModulePelvicAssistMeasure1;
    torch::jit::script::Module m_ModulePelvicAssistMeasure2;

    int m_ModuleFetusSeg1stLoadError;
    int m_ModuleFetusSeg2ndLoadError;
    int m_ModuleCNSSegLoadError;
    int m_ModuleSTInpaintingLoadError;
    int m_ModuleSTEnhancementLoadError;
    int m_ModuleSTAutoRestoreLoadError;
    int m_ModuleSTFaceClassificationLoadError;
    int m_ModuleCNSMTCLoadError;
    int m_ModuleCNSMTTLoadError;
    int m_ModuleCNSMTVLoadError;
    int m_ModuleNTSegLoadError;
    int m_ModuleNTMeasureLoadError;
    int m_ModulePelvicAssistSegLoadError;
    int m_ModulePelvicAssistMeasureLoadError1;
    int m_ModulePelvicAssistMeasureLoadError2;

    const int labelNum_NT = 7;
    int labelOffsetTable_NT[7] = { 0, 2, 3, 4, 5, 6, 7 };
    float labelThreshold_NT[7] = { 0.5,0.5,0.5,0.5,0.5,0.5,0.5 };
    int labelPriorityTable_NT[7] = { 0, 0, 0, 0, 0, 0, 0 };
    int processOnLabel_NT[7] = { 0, 0, 0, 0, 0, 0, 0 };
    int interpolationOrder_NT[7] = { 0, 0, 0, 0, 0, 0, 0 };

    int LoadModel(const std::wstring& filename, torch::jit::script::Module* module, int DeviceMode);
    int CheckModelLoaded(int CheckModel);
};

