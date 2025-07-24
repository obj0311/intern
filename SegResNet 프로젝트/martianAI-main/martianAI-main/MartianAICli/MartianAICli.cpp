// MartianAICli.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <Windows.h>
#include <string>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include "pch.h"
#include "MartianAIControl.h"
#include "..\MartianAI\MartianAIVar.h"
#include "atlbase.h"
#include "atlstr.h"

using namespace std;

unsigned char* m_InputBuffer = new unsigned char[256 * 256 * 256];
unsigned char* m_OutputBuffer = new unsigned char[256 * 256 * 256 * 2];

string usage = "usage: MartianAICli [-i|--inferencemode] inferencemode [-s|--size] size [-c|--cpu] [-g|--gpuid] gpuid [-f|--file] <input_file>\n";

void SetInferenceInfo(InferenceInfo* inferenceInfo) {
	if (inferenceInfo->InferenceMode == MARTIAN_SEG_PELVICASSIST) {
		inferenceInfo->labelNum = 6;
		inferenceInfo->morphologyMode = 0;
		// Dilation - 1, Erosion - 2, Both - 3

		// processOnTable
		inferenceInfo->processOnTable[0] = 0;        // BG 
		inferenceInfo->processOnTable[1] = 1;        // Anus
		inferenceInfo->processOnTable[2] = 1;        // Vagina
		inferenceInfo->processOnTable[3] = 1;        // Urethra
		inferenceInfo->processOnTable[4] = 1;        // PelvicBrim
		inferenceInfo->processOnTable[5] = 1;        // Blader

		// thresholdTable
		inferenceInfo->thresholdTable[0] = 0.0;      // BG
		inferenceInfo->thresholdTable[1] = 0.5;      // Anus
		inferenceInfo->thresholdTable[2] = 0.5;      // Vagina
		inferenceInfo->thresholdTable[3] = 0.5;      // Urethra
		inferenceInfo->thresholdTable[4] = 0.5;      // PelvicBrim
		inferenceInfo->thresholdTable[5] = 0.5;      // Blader
	}
}

wchar_t* GetCmdOption(wchar_t** begin, wchar_t** end, const std::wstring& option)
{
	wchar_t** itr = std::find(begin, end, option);
	if (itr != end && ++itr != end)
	{
		return *itr;
	}
	return 0;
}

bool CmdOptionExists(wchar_t** begin, wchar_t** end, const std::wstring& option)
{
	return std::find(begin, end, option) != end;
}

int wmain(int argc, wchar_t **argv)
{
	int returnValue = 0;

	try {
		int deviceMode = 0;  // 0 gpu , 1 cpu
		int gpuSelection = 0;
		int inferencemode = 0;
		int size = 0;
		wchar_t* infile;
		CString outputPath{ "./OutData/outputData_3DVolume.bin" };

		if (CmdOptionExists(argv, argv + argc, L"-c") || CmdOptionExists(argv, argv + argc, L"--cpu")) {
			deviceMode = 1;
			if (CmdOptionExists(argv, argv + argc, L"-g")|| CmdOptionExists(argv, argv + argc, L"--gpuid")) {
				throw std::invalid_argument("argument -c and -g are conflict");
			}
		}
		else {
			if (CmdOptionExists(argv, argv + argc, L"-g")) {
				gpuSelection = stoi(GetCmdOption(argv, argv + argc, L"-g"));
			}
			else if (CmdOptionExists(argv, argv + argc, L"--gpuid")) {
				gpuSelection = stoi(GetCmdOption(argv, argv + argc, L"--gpuid"));
			}
		}
		if (CmdOptionExists(argv, argv + argc, L"-i")) {
			inferencemode = (MARTIAN_INFERERNCE_MODE)stoi(GetCmdOption(argv, argv + argc, L"-i"));
		} else if (CmdOptionExists(argv, argv + argc, L"--inferencemode")) {
			inferencemode = (MARTIAN_INFERERNCE_MODE)stoi(GetCmdOption(argv, argv + argc, L"--inferencemode"));
		}
		else {
			throw std::invalid_argument("argument -i or --inferencemode is required");
		}
		if (CmdOptionExists(argv, argv + argc, L"-s")) {
			size = stoi(GetCmdOption(argv, argv + argc, L"-s"));
		}
		else if (CmdOptionExists(argv, argv + argc, L"--size")) {
			size = stoi(GetCmdOption(argv, argv + argc, L"--size"));
		}
		else {
			throw std::invalid_argument("argument -s or --size is required");
		}
		if (CmdOptionExists(argv, argv + argc, L"-f")) {
			infile = GetCmdOption(argv, argv + argc, L"-f");
		}
		else if (CmdOptionExists(argv, argv + argc, L"--file")) {
			infile = GetCmdOption(argv, argv + argc, L"--file");
		}
		else {
			throw std::invalid_argument("argument -f or --file is required");
		}
		CString inputFile(infile);
		
		std::string mpath = "./";
		std::wstring widestr = std::wstring(mpath.begin(), mpath.end());
		WCHAR modelPath[1024] = { 0, };
		wmemcpy(modelPath, widestr.c_str(), widestr.size());
		

		std::cout << "###########################################################\n";
		std::cout << "# Device Mode:" << deviceMode << " [ 0:GPU , 1:CPU ]\n";
		if (deviceMode == 0) {
			std::cout << "# GPU_ID:" << gpuSelection << "\n";
		}
		std::cout << "# Inference Mode:" << inferencemode << "\n";
		std::cout << "# Size:" << size << "\n";
		std::wcout << "# Model Path:" << modelPath << "\n";
		std::wcout << "# Input File:" << inputFile.GetString() << "\n";
		std::wcout << "# Output File:" << outputPath.GetString() << "\n";
		std::cout << "###########################################################\n";

		// Init
		MartianAIControl* martianAIControl = new MartianAIControl();
		returnValue = martianAIControl->Init(deviceMode, gpuSelection, modelPath);
		if (returnValue != 0) {
			std::cout << "ERROR in init :" << returnValue << "\n";
			return returnValue;
		}
		std::cout << "Init Done." << "\n";
		
		// CheckInit
		returnValue = martianAIControl->CheckInitDone(inferencemode);
		if (returnValue != 0) {
			std::cout << "ERROR in Init :" << returnValue << "\n";
			return returnValue;
		}
		std::cout << "CheckInitDone Done." << "\n";
		
		// Load File
		
		std::ifstream fin(inputFile, std::ios::binary);
		if (!fin)
		{
			std::wcout << "ERROR in Load File (" << inputFile.GetString() << ")\n";
			return EXIT_FAILURE;
		}
		fin.read(reinterpret_cast<char*>(m_InputBuffer), (std::streamsize)pow(size,3) * sizeof(unsigned char));
		std::wcout << "Load file Done.[" << inputFile.GetString() << "]" << "\n";
		
		// Inference
		InferenceInfo inferenceInfo;
		inferenceInfo.InferenceMode = inferencemode;
		inferenceInfo.xSize = size;
		inferenceInfo.ySize = size;
		inferenceInfo.zSize = size;
		memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
		memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
		
		SetInferenceInfo(&inferenceInfo);
		std::cout << "Inference Start.\n";
		int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);
		if (resultValue != 0) {
			std::cout << "ERROR in Inference(" << inferencemode << ") :" << returnValue << "\n";
			return EXIT_FAILURE;
		}
		std::cout << "Inference Done (" << inferenceInfo.inferenceTime << "ms)\n";

		// Store output
		
		std::ofstream FILE(outputPath, std::ios::out | std::ofstream::binary);
		FILE.write(reinterpret_cast<const char*>(m_OutputBuffer), inferenceInfo.xSize * inferenceInfo.ySize * (inferenceInfo.zSize) * sizeof(unsigned char));
		FILE.close();
		std::wcout << "save output Done (" << outputPath.GetString() << ")\n";
	}
	catch (const std::exception& x) {
		std::cerr << "MartianAICli: " << x.what() << '\n';
		std::cerr << usage;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
