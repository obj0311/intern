#include <queue>
#include <cmath>
#include <algorithm>
#include <vector>
#include "InferenceControl.h"
#include "US3DLog.h"
#include <Shlwapi.h>

using namespace std;

torch::NoGradGuard no_grad;							// Very Important

InferenceControl::InferenceControl(void)
{
	if (m_InputBuffer == NULL)
	{
		m_InputBuffer = new unsigned char[256 * 256 * 256];
		memset(m_InputBuffer, 0x00, sizeof(unsigned char) * 256 * 256 * 256);
	}

	if (m_OutputBuffer == NULL)
	{
		m_OutputBuffer = new unsigned char[256 * 256 * 256 * 2];
		memset(m_OutputBuffer, 0x00, sizeof(unsigned char) * 256 * 256 * 256 * 2);
	}

	if (m_ControlSharedMemory == NULL)
	{
		LOG_T("ControlSharedMemory");
		m_ControlSharedMemory = new ControlSharedMemory();
	}

	torch::jit::setGraphExecutorOptimize(false);
	torch::NoGradGuard no_grad;							// Very Important

	isCPUOnly = 0;
	m_ModuleFetusSeg1stLoadError = 1;
	m_ModuleFetusSeg2ndLoadError = 1;
	m_ModuleCNSSegLoadError = 1;
	m_ModuleSTInpaintingLoadError = 1;
	m_ModuleSTEnhancementLoadError = 1;
	m_ModuleSTAutoRestoreLoadError = 1;
	m_ModuleCNSMTCLoadError = 1;
	m_ModuleCNSMTTLoadError = 1;
	m_ModuleCNSMTVLoadError = 1;
	m_ModulePelvicAssistSegLoadError = 1;
	m_ModulePelvicAssistMeasureLoadError1 = 1;
	m_ModuleNTSegLoadError = 1;
	m_ModuleNTMeasureLoadError = 1;
}

InferenceControl::~InferenceControl()
{
	delete mCommonProcess;
	delete mBeautify;
	delete mFetusSegmentation;
	delete mCNSAuto;
	delete mPelvicAssist;
	delete[] m_InputBuffer;
	delete[] m_OutputBuffer;
	delete m_ControlSharedMemory;
}

int InferenceControl::InitInference(int loadMode, int gpuSelection, int ModelPathLen, WCHAR* ModelPath)
{
	try
	{
		if (InitFlag == 0)
		{
			std::chrono::steady_clock::time_point start_Init = std::chrono::steady_clock::now();

			char deviceBuffer[1024];
			sprintf_s(deviceBuffer, sizeof(deviceBuffer), "cuda:%d", gpuSelection);
			mDeviceString = deviceBuffer;
			
			LOG_T("InitInference");

			std::wstring widestr(ModelPath);
			std::string str(widestr.begin(), widestr.end());

			LOG_T("Model Path({})", str);

			WCHAR loadOptionIniPath[MAX_PATH] = { 0, };
			if (GetModuleFileNameW(GetModuleHandleA(NULL), loadOptionIniPath, MAX_PATH) < MAX_PATH)
			{
				PathRemoveFileSpecW(loadOptionIniPath);
				PathAppendW(loadOptionIniPath, L"MartianLoadOption.ini");
			}

			WCHAR wcLoadMode[MAX_PATH] = { 0, };
			wsprintfW(wcLoadMode, L"%s%d", L"mode", loadMode);
			WCHAR wcTempModelPath[MAX_PATH] = { 0, };

			int initMode = GetPrivateProfileIntW(wcLoadMode, L"initMode", 0, loadOptionIniPath);

			GetPrivateProfileString(wcLoadMode, L"f1path", L"", wcTempModelPath, 260, loadOptionIniPath);
			m_ModelFileFetusSeg1st = widestr + std::wstring(wcTempModelPath);

			GetPrivateProfileString(wcLoadMode, L"f2path", L"", wcTempModelPath, 260, loadOptionIniPath);
			m_ModelFileFetusSeg2nd = widestr + std::wstring(wcTempModelPath);

			GetPrivateProfileString(wcLoadMode, L"s1path", L"", wcTempModelPath, 260, loadOptionIniPath);
			m_ModelFileSTInpainting = widestr + std::wstring(wcTempModelPath);

			GetPrivateProfileString(wcLoadMode, L"s2path", L"", wcTempModelPath, 260, loadOptionIniPath);
			m_ModelFileSTEnhancement = widestr + std::wstring(wcTempModelPath);

			GetPrivateProfileString(wcLoadMode, L"s3path", L"", wcTempModelPath, 260, loadOptionIniPath);
			m_ModelFileSTAutoRestore = widestr + std::wstring(wcTempModelPath);

			GetPrivateProfileString(wcLoadMode, L"s4path", L"", wcTempModelPath, 260, loadOptionIniPath);
			m_ModelFileSTFaceClassification = widestr + std::wstring(wcTempModelPath);

			GetPrivateProfileString(wcLoadMode, L"c1path", L"", wcTempModelPath, 260, loadOptionIniPath);
			m_ModelFileCNSSeg = widestr + std::wstring(wcTempModelPath);

			GetPrivateProfileString(wcLoadMode, L"c2path", L"", wcTempModelPath, 260, loadOptionIniPath);
			m_ModelFileCNSMTC = widestr + std::wstring(wcTempModelPath);

			GetPrivateProfileString(wcLoadMode, L"c3path", L"", wcTempModelPath, 260, loadOptionIniPath);
			m_ModelFileCNSMTT = widestr + std::wstring(wcTempModelPath);

			GetPrivateProfileString(wcLoadMode, L"c4path", L"", wcTempModelPath, 260, loadOptionIniPath);
			m_ModelFileCNSMTV = widestr + std::wstring(wcTempModelPath);

			GetPrivateProfileString(wcLoadMode, L"p1path", L"", wcTempModelPath, 260, loadOptionIniPath);
			m_ModelFilePelvicAssistSeg = widestr + std::wstring(wcTempModelPath);

			GetPrivateProfileString(wcLoadMode, L"p2path", L"", wcTempModelPath, 260, loadOptionIniPath);
			m_ModelFilePelvicAssistMeasure1 = widestr + std::wstring(wcTempModelPath);

			GetPrivateProfileString(wcLoadMode, L"n1path", L"", wcTempModelPath, 260, loadOptionIniPath);
			m_ModelFileNTSeg_head = widestr + std::wstring(wcTempModelPath);

			GetPrivateProfileString(wcLoadMode, L"n2path", L"", wcTempModelPath, 260, loadOptionIniPath);
			m_ModelFileNTSeg_inside = widestr + std::wstring(wcTempModelPath);

			GetPrivateProfileString(wcLoadMode, L"n3path", L"", wcTempModelPath, 260, loadOptionIniPath);
			m_ModelFileNTMeasure = widestr + std::wstring(wcTempModelPath);

			m_ModuleFetusSeg1stLoadError = LoadModel(m_ModelFileFetusSeg1st, &m_ModuleFetusSeg1st, initMode);
			m_ModuleFetusSeg2ndLoadError = LoadModel(m_ModelFileFetusSeg2nd, &m_ModuleFetusSeg2nd, initMode);
			m_ModuleSTInpaintingLoadError = LoadModel(m_ModelFileSTInpainting, &m_ModuleSTInpainting, initMode);
			m_ModuleSTEnhancementLoadError = LoadModel(m_ModelFileSTEnhancement, &m_ModuleSTEnhancement, initMode);
			m_ModuleSTAutoRestoreLoadError = LoadModel(m_ModelFileSTAutoRestore, &m_ModuleSTAutoRestore, initMode);
			m_ModuleSTFaceClassificationLoadError = LoadModel(m_ModelFileSTFaceClassification, &m_ModuleSTFaceClassification, initMode);
			m_ModuleCNSSegLoadError = LoadModel(m_ModelFileCNSSeg, &m_ModuleCNSSeg, initMode);
			m_ModuleCNSMTCLoadError = LoadModel(m_ModelFileCNSMTC, &m_ModuleCNSMTC, initMode);
			m_ModuleCNSMTTLoadError = LoadModel(m_ModelFileCNSMTT, &m_ModuleCNSMTT, initMode);
			m_ModuleCNSMTVLoadError = LoadModel(m_ModelFileCNSMTV, &m_ModuleCNSMTV, initMode);
			m_ModuleNTSegLoadError = LoadModel(m_ModelFileNTSeg_head, &m_ModuleNTSeg_head, initMode);
			m_ModuleNTSegLoadError = LoadModel(m_ModelFileNTSeg_inside, &m_ModuleNTSeg_inside, initMode);
			m_ModuleNTMeasureLoadError = LoadModel(m_ModelFileNTMeasure, &m_ModuleNTMeasure, initMode);
			m_ModulePelvicAssistSegLoadError = LoadModel(m_ModelFilePelvicAssistSeg, &m_ModulePelvicAssistSeg, initMode);
			m_ModulePelvicAssistMeasureLoadError1 = LoadModel(m_ModelFilePelvicAssistMeasure1, &m_ModulePelvicAssistMeasure1, initMode);

			if (initMode == 0)
			{
				LOG_T("DeviceMode = GPU ({})", mDeviceString);

				InitFlag = 1;
				isCPUOnly = 0;
				// Warm-up
				std::vector<torch::jit::IValue> inputs;
				inputs.push_back(torch::rand({ 1, 1, 128, 128, 128 }).to(mDeviceString));

				if (m_ModuleFetusSeg1stLoadError == RET_OK)
				{
					auto output = m_ModuleFetusSeg1st.forward(inputs).toTensor().to(at::kCPU);
				}
				if (m_ModuleCNSSegLoadError == RET_OK)
				{
					auto output = m_ModuleCNSSeg.forward(inputs).toTensor().to(at::kCPU);
				}
				if (m_ModulePelvicAssistSegLoadError == RET_OK)
				{
					auto output = m_ModulePelvicAssistSeg.forward(inputs).toTensor().to(at::kCPU);
				}
				c10::cuda::CUDACachingAllocator::emptyCache();
			}
			else
			{
				LOG_T("DeviceMode = CPU");

				InitFlag = 1;
				isCPUOnly = 1;
			}

			mCommonProcess = new CommonProcess(isCPUOnly, mDeviceString);
			mBeautify = new Beautify(isCPUOnly, mDeviceString);
			mCNSAuto = new CNSAuto(isCPUOnly, mDeviceString, mCommonProcess);
			mFetusSegmentation = new FetusSegmentation(isCPUOnly, mDeviceString, mCommonProcess);
			mPelvicAssist = new PelvicAssist(isCPUOnly, mDeviceString);
			mNTAuto = new NTAuto(isCPUOnly, mDeviceString, mCommonProcess);

			std::chrono::steady_clock::time_point end_Init = std::chrono::steady_clock::now();
			int initTime = std::chrono::duration_cast<std::chrono::milliseconds>(end_Init - start_Init).count();
			LOG_T("MartianAI Total Init Time = {}ms", initTime);	
		}
		else
		{
			if (loadMode >= 10)
			{
				return CheckModelLoaded(loadMode);
			}
			else
			{
				LOG_T("Init Inference Pass");
			}
		}
	}
	catch (const c10::Error& e) {
		LOG_E("error in InitInference");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}

	if (m_ModuleFetusSeg1stLoadError == 0 || m_ModuleFetusSeg2ndLoadError == 0 ||
		m_ModuleCNSSegLoadError == 0 ||
		m_ModuleSTInpaintingLoadError == 0 || m_ModuleSTEnhancementLoadError == 0 || m_ModuleSTAutoRestoreLoadError == 0 ||
		m_ModuleCNSMTCLoadError == 0 || m_ModuleCNSMTTLoadError == 0 || m_ModuleCNSMTVLoadError == 0 || m_ModuleNTSegLoadError == 0 || m_ModuleNTMeasureLoadError == 0 ||
		m_ModulePelvicAssistSegLoadError == 0 || m_ModulePelvicAssistMeasureLoadError1 == 0)
	{
		return RET_OK;
	}
	else
	{
		return RET_FAILURE;
	}
}

int InferenceControl::LoadModel(const std::wstring& filename, torch::jit::script::Module* module, int DeviceMode)
{
	unsigned char Key[] = { 0x76, 0x6f, 0x6c, 0x75, 0x6d, 0x65, 0x74, 0x72, 0x69, 0x63, 0x20, 0x61, 0x69, 0x6c, 0x61, 0x62 };
	unsigned char expandedKey[176] = { 0, };
	KeyExpansion(Key, expandedKey);

	std::ifstream fin(filename, std::ios::binary);

	if (!fin)
	{
		LOG_T(L"fail to LoadModel {}", filename);
		return RET_FAILURE;
	}
	else
	{
		std::chrono::steady_clock::time_point start_dec = std::chrono::steady_clock::now();

		fin.seekg(0, ios::end);
		size_t length_load = fin.tellg();
		fin.seekg(0, ios::beg);
		unsigned char* modelBin = new unsigned char[length_load];

		fin.read(reinterpret_cast<char*>(modelBin), length_load * sizeof(unsigned char));

		int encryptionSize = ENCRYPTION_SIZE;
		if (length_load < encryptionSize)
		{
			encryptionSize = 16 * (int)(length_load / 16);
		}

		for (int i = 0; i < encryptionSize / 2; i += 16) {
			AESDecrypt(modelBin + i, expandedKey, modelBin + i);
		}
		for (int i = length_load - 16; i > (length_load - encryptionSize / 2); i -= 16) {
			AESDecrypt(modelBin + i, expandedKey, modelBin + i);
		}

		std::vector< char> retval(modelBin, modelBin + length_load);

		std::istringstream iss(std::string(retval.begin(), retval.end()));

		std::chrono::steady_clock::time_point end_dec = std::chrono::steady_clock::now();
		int DecryptionTime = std::chrono::duration_cast<std::chrono::milliseconds>(end_dec - start_dec).count();
		LOG_T(L"{} Decryption Time = {}ms", filename, DecryptionTime);

		delete[] modelBin;

		try
		{
			if (DeviceMode == 0)
			{
				*module = torch::jit::load(iss, torch::kCUDA);
				module->to(mDeviceString);
				module->eval();
			}
			else
			{
				*module = torch::jit::load(iss, torch::kCPU);
				module->eval();
			}
		}
		catch (const c10::Error& e) {
			LOG_E("fail in LoadModel");
			LOG_E("Message : {}", e.msg());
			return RET_FAILURE;
		}
	}

	return RET_OK;
}

int InferenceControl::CheckModelLoaded(int CheckModel)
{
	switch (CheckModel)
	{
	case MARTIAN_SEG_FETUS_1ST:
		return m_ModuleFetusSeg1stLoadError;
	case MARTIAN_SEG_FETUS_2ND:
		return m_ModuleFetusSeg2ndLoadError;
	case MARTIAN_STYLE_TRANSFER_INPAINTING:
		return m_ModuleSTInpaintingLoadError;
	case MARTIAN_STYLE_TRANSFER_ENHANCEMENT:
		return m_ModuleSTEnhancementLoadError;
	case MARTIAN_STYLE_TRANSFER_AUTORESTORE:
		return m_ModuleSTAutoRestoreLoadError;
	case MARTIAN_STYLE_TRANSFER_FACECLASSIFICATION:
		return m_ModuleSTFaceClassificationLoadError;
	case MARTIAN_CNS_SEG:
		return m_ModuleCNSSegLoadError;
	case MARTIAN_CNS_MEASURETC:
		return m_ModuleCNSMTCLoadError;
	case MARTIAN_CNS_MEASURETT:
		return m_ModuleCNSMTTLoadError;
	case MARTIAN_CNS_MEASURETV:
		return m_ModuleCNSMTVLoadError;
	case MARTIAN_SEG_NT:
		return m_ModuleNTSegLoadError;
	case MARTIAN_SEG_NTMEASURE:
		return m_ModuleNTMeasureLoadError;
	case MARTIAN_SEG_PELVICASSIST:
		return m_ModulePelvicAssistSegLoadError;
	case MARTIAN_SEG_PELVICMEASURE:
		return m_ModulePelvicAssistMeasureLoadError1;
	default:
		return 1;
	}
}

int InferenceControl::RunInference(int Size, int Mode)
{
	InferenceInfo inferenceInfo;
	std::chrono::steady_clock::time_point begin_Inference = std::chrono::steady_clock::now();

	int resultValue = true;
	LOG_T("RunInference");

	if (m_ControlSharedMemory->ReadDataFromSharedMemory(m_InputBuffer, inferenceInfo))
	{
		if (Mode == MARTIAN_SEG_FETUS_1ST)			//1st Fetus
		{
			if (m_ModuleFetusSeg1stLoadError == 0)
			{
				std::vector<unsigned char> fdata(m_InputBuffer, m_InputBuffer + Size);
				at::Tensor InDataTensor = torch::tensor(fdata);
				InDataTensor = InDataTensor.to(torch::kFloat32);

				resultValue = mFetusSegmentation->ExcuteFetusSegmentation(m_ModuleFetusSeg1st, InDataTensor, Mode, &inferenceInfo, m_OutputBuffer);
			}
			else
			{
				resultValue = RET_FAILURE;
			}
		}
		else if (Mode == MARTIAN_SEG_FETUS_2ND)		//2nd Fetus
		{
			if (m_ModuleFetusSeg2ndLoadError == 0)
			{
				std::vector<unsigned char> fdata(m_InputBuffer, m_InputBuffer + Size);
				at::Tensor InDataTensor = torch::tensor(fdata);
				InDataTensor = InDataTensor.to(torch::kFloat32);

				resultValue = mFetusSegmentation->ExcuteFetusSegmentation(m_ModuleFetusSeg2nd, InDataTensor, Mode, &inferenceInfo, m_OutputBuffer);
			}
			else
			{
				resultValue = RET_FAILURE;
			}
		}
		else if (Mode == MARTIAN_CNS_SEG)		//5D CNS
		{
			if (m_ModuleCNSSegLoadError == 0)
			{
				inferenceInfo.errorCode = 0;
				inferenceInfo.returnValue = 0;

				std::vector<unsigned char> fdata(m_InputBuffer, m_InputBuffer + Size);
				at::Tensor InDataTensor = torch::tensor(fdata);
				InDataTensor = InDataTensor.to(torch::kFloat32);

				resultValue = mCNSAuto->ExcuteCNSAuto(m_ModuleCNSSeg, InDataTensor, &inferenceInfo, m_OutputBuffer);
			}
			else
			{
				resultValue = RET_FAILURE;
			}
		}
		else if (Mode == MARTIAN_CNS_MEASURETC)
		{
			if (m_ModuleCNSMTCLoadError == 0)
			{
				std::vector<unsigned char> fdata(m_InputBuffer, m_InputBuffer + Size);
				at::Tensor InDataTensor = torch::tensor(fdata);
				InDataTensor = InDataTensor.to(torch::kFloat32);

				resultValue = mCNSAuto->mmSeg(m_ModuleCNSMTC, InDataTensor, InDataTensor, inferenceInfo.xSize, inferenceInfo.ySize, inferenceInfo.ySize);

				InDataTensor = InDataTensor.to(torch::kUInt8);
				inferenceInfo.zSize = 1;

				unsigned char* outputfData = InDataTensor.flatten().data_ptr<unsigned char>();
				memcpy(m_OutputBuffer, outputfData, (size_t)inferenceInfo.xSize* (size_t)inferenceInfo.ySize * sizeof(unsigned char));
			}
			else
			{
				resultValue = RET_FAILURE;
			}
		}
		else if (Mode == MARTIAN_CNS_MEASURETT)
		{
			if (m_ModuleCNSMTTLoadError == 0)
			{
				std::vector<unsigned char> fdata(m_InputBuffer, m_InputBuffer + Size);
				at::Tensor InDataTensor = torch::tensor(fdata);
				InDataTensor = InDataTensor.to(torch::kFloat32);

				resultValue = mCNSAuto->mmSeg(m_ModuleCNSMTT, InDataTensor, InDataTensor, inferenceInfo.xSize, inferenceInfo.ySize, inferenceInfo.ySize);

				InDataTensor = InDataTensor.to(torch::kUInt8);
				inferenceInfo.zSize = 1;

				unsigned char* outputfData = InDataTensor.flatten().data_ptr<unsigned char>();
				memcpy(m_OutputBuffer, outputfData, (size_t)inferenceInfo.xSize * (size_t)inferenceInfo.ySize * sizeof(unsigned char));
			}
			else
			{
				resultValue = RET_FAILURE;
			}
		}
		else if (Mode == MARTIAN_CNS_MEASURETV)
		{
			if (m_ModuleCNSMTVLoadError == 0)
			{
				std::vector<unsigned char> fdata(m_InputBuffer, m_InputBuffer + Size);
				at::Tensor InDataTensor = torch::tensor(fdata);
				InDataTensor = InDataTensor.to(torch::kFloat32);

				resultValue = mCNSAuto->mmSeg(m_ModuleCNSMTV, InDataTensor, InDataTensor, inferenceInfo.xSize, inferenceInfo.ySize, inferenceInfo.ySize);

				InDataTensor = InDataTensor.to(torch::kUInt8);
				inferenceInfo.zSize = 1;

				unsigned char* outputfData = InDataTensor.flatten().data_ptr<unsigned char>();
				memcpy(m_OutputBuffer, outputfData, (size_t)inferenceInfo.xSize * (size_t)inferenceInfo.ySize * sizeof(unsigned char));
			}
			else
			{
				resultValue = RET_FAILURE;
			}
		}
		else if (Mode == MARTIAN_STYLE_TRANSFER_INPAINTING)		//Style Transfer
		{
			if (m_ModuleSTInpaintingLoadError == 0)
			{
				std::vector<unsigned char> fdata(m_InputBuffer, m_InputBuffer + Size);
				at::Tensor InDataTensor = torch::tensor(fdata);
				InDataTensor = InDataTensor.to(torch::kFloat32);

				resultValue = mBeautify->StyleTransfer(m_ModuleSTInpainting, InDataTensor, InDataTensor, inferenceInfo.xSize, inferenceInfo.ySize, inferenceInfo.imageRotate);
				InDataTensor = InDataTensor.to(torch::kUInt8);

				unsigned char* outputfData = InDataTensor.flatten().data_ptr<unsigned char>();
				memcpy(m_OutputBuffer, outputfData, (size_t)inferenceInfo.xSize * (size_t)inferenceInfo.ySize * 3 * sizeof(unsigned char));
			}
			else
			{
				resultValue = RET_FAILURE;
			}
		}
		else if (Mode == MARTIAN_STYLE_TRANSFER_ENHANCEMENT)		//Style Transfer
		{
			if (m_ModuleSTEnhancementLoadError == 0)
			{
				std::vector<unsigned char> fdata(m_InputBuffer, m_InputBuffer + Size);
				at::Tensor InDataTensor = torch::tensor(fdata);
				InDataTensor = InDataTensor.to(torch::kFloat32);

				resultValue = mBeautify->StyleTransfer(m_ModuleSTEnhancement, InDataTensor, InDataTensor, inferenceInfo.xSize, inferenceInfo.ySize, inferenceInfo.imageRotate);
				InDataTensor = InDataTensor.to(torch::kUInt8);

				unsigned char* outputfData = InDataTensor.flatten().data_ptr<unsigned char>();
				memcpy(m_OutputBuffer, outputfData, (size_t)inferenceInfo.xSize * (size_t)inferenceInfo.ySize * 3 * sizeof(unsigned char));
			}
		}
		else if (Mode == MARTIAN_STYLE_TRANSFER_AUTORESTORE)		//Style Transfer
		{
			if (m_ModuleSTAutoRestoreLoadError == 0)
			{
				std::vector<unsigned char> fdata(m_InputBuffer, m_InputBuffer + Size);
				at::Tensor InDataTensor = torch::tensor(fdata);
				InDataTensor = InDataTensor.to(torch::kFloat32);

				resultValue = mBeautify->StyleTransfer(m_ModuleSTAutoRestore, InDataTensor, InDataTensor, inferenceInfo.xSize, inferenceInfo.ySize, inferenceInfo.imageRotate);
				InDataTensor = InDataTensor.to(torch::kUInt8);

				unsigned char* outputfData = InDataTensor.flatten().data_ptr<unsigned char>();
				memcpy(m_OutputBuffer, outputfData, (size_t)inferenceInfo.xSize * (size_t)inferenceInfo.ySize * 3 * sizeof(unsigned char));
			}
		}
		else if (Mode == MARTIAN_STYLE_TRANSFER_FACECLASSIFICATION)		//Style Transfer
		{
			if (m_ModuleSTFaceClassificationLoadError == 0)
			{
				inferenceInfo.returnValue = 0;

				std::vector<unsigned char> fdata(m_InputBuffer, m_InputBuffer + Size);
				at::Tensor InDataTensor = torch::tensor(fdata);
				InDataTensor = InDataTensor.to(torch::kFloat32);

				resultValue = mBeautify->FaceClassification(m_ModuleSTFaceClassification, InDataTensor, InDataTensor, inferenceInfo.xSize, inferenceInfo.ySize, &inferenceInfo.errorCode);
			}
		}
		else if (Mode == MARTIAN_SEG_NT)            //5D NT
		{
			if (m_ModuleNTSegLoadError == 0)
			{
				inferenceInfo.errorCode = 0;
				inferenceInfo.returnValue = 0;

				std::vector<unsigned char> fdata(m_InputBuffer, m_InputBuffer + Size);
				at::Tensor InDataTensor = torch::tensor(fdata);
				InDataTensor = InDataTensor.to(torch::kFloat32);

				resultValue = mNTAuto->ExcuteNTAuto(m_ModuleNTSeg_head, m_ModuleNTSeg_inside, InDataTensor, &inferenceInfo, m_OutputBuffer);
			}
			else
			{
				resultValue = RET_FAILURE;
			}
		}
		else if (Mode == MARTIAN_SEG_NTMEASURE)            //5D NT
		{
			if (m_ModuleNTMeasureLoadError == 0)
			{
				inferenceInfo.errorCode = 0;
				inferenceInfo.returnValue = 0;

				std::vector<unsigned char> fdata(m_InputBuffer, m_InputBuffer + Size);
				at::Tensor InDataTensor = torch::tensor(fdata);
				InDataTensor = InDataTensor.to(torch::kFloat32);
				bool RGB = false;
				if (inferenceInfo.zSize == 3)
					RGB = true;
				resultValue = mNTAuto->Measure2DSagmentation(m_ModuleNTMeasure, InDataTensor, InDataTensor, inferenceInfo.pcaVector, inferenceInfo.xSize, inferenceInfo.ySize, RGB);
				
				if (resultValue != RET_FAILURE) {
					inferenceInfo.zSize = 2;
					unsigned char* outputfData = InDataTensor.flatten().data_ptr<unsigned char>();
					memcpy(m_OutputBuffer, outputfData, NT2DSEG_IMAGE_SIZE * NT2DSEG_IMAGE_SIZE * 2 * sizeof(unsigned char));
				}
			}
			else
			{
				resultValue = RET_FAILURE;
			}
		}
		else if (Mode == MARTIAN_SEG_PELVICASSIST)
		{
			if (m_ModulePelvicAssistSegLoadError == 0)
			{
				int size = pow(PELVICASSIST_SEG_OUT_VOLUME_SIZE,3) * sizeof(unsigned char);
				std::vector<unsigned char> fdata(m_InputBuffer, m_InputBuffer + Size);
				at::Tensor InDataTensor = torch::tensor(fdata);
				at::Tensor OutDataTensor;
				inferenceInfo.errorCode = 0;
				inferenceInfo.returnValue = 0;
				memset(inferenceInfo.SegMean, 0x00, sizeof(inferenceInfo.SegMean));
				memset(inferenceInfo.pcaMean, 0x00, sizeof(inferenceInfo.pcaMean));
				memset(inferenceInfo.pcaVector, 0x00 , sizeof(inferenceInfo.pcaVector));
				InDataTensor = InDataTensor.to(torch::kFloat32);
				if (inferenceInfo.xSize != inferenceInfo.ySize || inferenceInfo.xSize != inferenceInfo.zSize) {
					resultValue = RET_FAILURE;
				}
				else {
					resultValue = mPelvicAssist->segResnet(
						m_ModulePelvicAssistSeg,
						InDataTensor,
						OutDataTensor,
						inferenceInfo.SegMean,
						inferenceInfo.pcaVector,
						inferenceInfo.labelNum,
						inferenceInfo.thresholdTable,
						inferenceInfo.processOnTable,
						inferenceInfo.xSize,
						inferenceInfo.gaussianFilterMode,
						inferenceInfo.morphologyMode,
						&inferenceInfo.errorCode,
						&inferenceInfo.returnValue,
						m_OutputBuffer
					);
					inferenceInfo.xSize = PELVICASSIST_SEG_OUT_VOLUME_SIZE;
					inferenceInfo.ySize = PELVICASSIST_SEG_OUT_VOLUME_SIZE;
					inferenceInfo.zSize = PELVICASSIST_SEG_OUT_VOLUME_SIZE;
				}
			}
			else
			{
				resultValue = RET_FAILURE;
			}
		}
		else if (Mode == MARTIAN_SEG_PELVICMEASURE)
		{
			if (m_ModulePelvicAssistMeasureLoadError1 == 0)
			{
				std::vector<unsigned char> fdata(m_InputBuffer, m_InputBuffer + Size);
				at::Tensor InDataTensor = torch::tensor(fdata);
				InDataTensor = InDataTensor.to(torch::kFloat32);

				resultValue = mPelvicAssist->Measure2DSagmentation(m_ModulePelvicAssistMeasure1, InDataTensor, InDataTensor, inferenceInfo.xSize, inferenceInfo.ySize, inferenceInfo.imageRotate);

				InDataTensor = InDataTensor.to(torch::kUInt8);

				unsigned char* outputfData = InDataTensor.flatten().data_ptr<unsigned char>();
				memcpy(m_OutputBuffer, outputfData, (size_t)inferenceInfo.xSize* (size_t)inferenceInfo.ySize * 3 * sizeof(unsigned char));
			}
			else
			{
				resultValue = RET_FAILURE;
			}
			}
		else if (Mode == MARTIAN_3DPROC_GAUSSION)	//Gaussion Filtering
		{
			std::vector<unsigned char> fdata(m_InputBuffer, m_InputBuffer + Size);
			at::Tensor InDataTensor = torch::tensor(fdata);
			InDataTensor = InDataTensor.to(torch::kFloat32);

			InDataTensor = InDataTensor.reshape({ 1, 1, inferenceInfo.xSize, inferenceInfo.ySize, inferenceInfo.zSize });

			resultValue = mCommonProcess->gaussian3DFilter(InDataTensor, InDataTensor, inferenceInfo.kernelSize, 1);        // 1 x 1 x 3 x 3 x 3  Tensor

			InDataTensor = InDataTensor.to(torch::kUInt8);

			if (resultValue == 0)
			{
				unsigned char* outputfData = InDataTensor.flatten().data_ptr<unsigned char>();
				memcpy(m_OutputBuffer, outputfData, Size * sizeof(unsigned char));
			}
		}
		else if (Mode == MARTIAN_3DPROC_DILATION)	//Dilation Process
		{
			std::vector<unsigned char> fdata(m_InputBuffer, m_InputBuffer + Size);
			at::Tensor InDataTensor = torch::tensor(fdata);
			InDataTensor = InDataTensor.to(torch::kFloat32);

			at::Tensor dummyTensor = torch::zeros({ 1, inferenceInfo.labelNum, inferenceInfo.xSize, inferenceInfo.ySize, inferenceInfo.zSize });

			int labelPriorityTable[6] = { 0, 3, 14, 12, 5, 7 };

			resultValue = mCommonProcess->toSingleLabels(InDataTensor, dummyTensor, inferenceInfo.labelNum, labelPriorityTable, inferenceInfo.xSize, inferenceInfo.ySize, inferenceInfo.zSize);
			if (!resultValue)
			{
				resultValue = mCommonProcess->dilation3D(dummyTensor, dummyTensor, inferenceInfo.labelNum, inferenceInfo.xSize, inferenceInfo.ySize, inferenceInfo.zSize, inferenceInfo.processOnTable, labelPriorityTable, 1);
				if (!resultValue)
				{
					// Labeling
					for (int labelIndex = 0; labelIndex < inferenceInfo.labelNum; labelIndex++)
					{
						dummyTensor.index_put_({ 0,labelIndex, Slice(),Slice() ,Slice() },
							torch::mul(dummyTensor.index({ 0,labelIndex,Slice(),Slice() ,Slice() }), labelPriorityTable[labelIndex]));   // Label Offset
					}

					//Merge
					InDataTensor = torch::sum(dummyTensor, 1);

					InDataTensor = InDataTensor.to(torch::kUInt8);

					unsigned char* outputfData = InDataTensor.flatten().data_ptr<unsigned char>();
					memcpy(m_OutputBuffer, outputfData, Size * sizeof(unsigned char));
				}
			}
		}
		else if (Mode == MARTIAN_3DPROC_EROSION)	//Erosion Process
		{
			std::vector<unsigned char> fdata(m_InputBuffer, m_InputBuffer + Size);
			at::Tensor InDataTensor = torch::tensor(fdata);
			InDataTensor = InDataTensor.to(torch::kFloat32);

			at::Tensor dummyTensor = torch::zeros({ 1, inferenceInfo.labelNum, inferenceInfo.xSize, inferenceInfo.ySize, inferenceInfo.zSize });

			int labelPriorityTable[6] = { 0, 3, 14, 12, 5, 7 };

			resultValue = mCommonProcess->toSingleLabels(InDataTensor, dummyTensor, inferenceInfo.labelNum, labelPriorityTable, inferenceInfo.xSize, inferenceInfo.ySize, inferenceInfo.zSize);
			if (!resultValue)
			{
				resultValue = mCommonProcess->erosion3D(dummyTensor, dummyTensor, inferenceInfo.labelNum, inferenceInfo.xSize, inferenceInfo.ySize, inferenceInfo.zSize, inferenceInfo.processOnTable, 1);
				if (!resultValue)
				{
					// Labeling
					for (int labelIndex = 0; labelIndex < inferenceInfo.labelNum; labelIndex++)
					{
						dummyTensor.index_put_({ 0,labelIndex, Slice(),Slice() ,Slice() },
							torch::mul(dummyTensor.index({ 0,labelIndex,Slice(),Slice() ,Slice() }), labelPriorityTable[labelIndex]));   // Label Offset
					}

					//Merge
					InDataTensor = torch::sum(dummyTensor, 1);

					InDataTensor = InDataTensor.to(torch::kUInt8);

					unsigned char* outputfData = InDataTensor.flatten().data_ptr<unsigned char>();
					memcpy(m_OutputBuffer, outputfData, Size * sizeof(unsigned char));
				}
			}
		}
		else if (Mode == MARTIAN_PROC_PCA)	//PCA Analysis
		{
			std::vector<unsigned char> fdata(m_InputBuffer, m_InputBuffer + Size);
			at::Tensor InDataTensor = torch::tensor(fdata);
			InDataTensor = InDataTensor.to(torch::kFloat32);

			at::Tensor eVec;
			at::Tensor sampleMean;
			at::Tensor eValue;
			resultValue = mCommonProcess->torchPCA(InDataTensor, inferenceInfo.pcaTargetLabel, eVec, sampleMean, eValue, inferenceInfo.xSize, inferenceInfo.ySize, inferenceInfo.zSize);

			if (resultValue == RET_OK)
			{
				eVec = eVec.flatten();		// {3,3}의 경우 먼저 flatten 한번 해줘야 한다.
				float* outputEVec = eVec.flatten().data_ptr<float>();
				float* outputSampleMean = sampleMean.flatten().data_ptr<float>();

				memcpy(inferenceInfo.pcaVector, outputEVec, 9 * sizeof(float));
				memcpy(inferenceInfo.pcaMean, outputSampleMean, 3 * sizeof(float));
			}
		}
		else
		{
			resultValue = RET_FAILURE;
		}

		c10::cuda::CUDACachingAllocator::emptyCache();
		
		std::chrono::steady_clock::time_point end_Inference = std::chrono::steady_clock::now();
		inferenceInfo.inferenceTime = std::chrono::duration_cast<std::chrono::milliseconds>(end_Inference - begin_Inference).count();
		LOG_I("MartianAI Total time = {}ms (Mode = {})", inferenceInfo.inferenceTime, Mode);

		if (resultValue == RET_OK || resultValue == RET_CHECK_ERROR_CODE)
		{
			if (m_ControlSharedMemory->WriteDataToSharedMemory(m_OutputBuffer, inferenceInfo))
			{
				resultValue = RET_OK;
			}
		}
	}
	return resultValue;
}

