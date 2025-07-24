#include "pch.h"
#include "CppUnitTest.h"
#include "MartianAIControl.h"
#include <fstream>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

enum MARTIAN_INFERERNCE_MODE
{
	MARTIAN_SEG_FETUS_1ST = 10,
	MARTIAN_SEG_FETUS_2ND,
	MARTIAN_STYLE_TRANSFER_INPAINTING = 20,
	MARTIAN_STYLE_TRANSFER_AUTORESTORE,
	MARTIAN_STYLE_TRANSFER_ENHANCEMENT,
	MARTIAN_STYLE_TRANSFER_FACECLASSIFICATION,
	MARTIAN_CNS_SEG = 30,
	MARTIAN_CNS_MEASURETC = 31,
	MARTIAN_CNS_MEASURETT,
	MARTIAN_CNS_MEASURETV,
	MARTIAN_SEG_NT = 40,
	MARTIAN_SEG_NTMEASURE,
	MARTIAN_SEG_PELVICASSIST = 60,
	MARTIAN_SEG_PELVICMEASURE1, 
	MARTIAN_SEG_PELVICMEASURE2,
	MARTIAN_3DPROC_GAUSSION = 110,
	MARTIAN_3DPROC_DILATION,
	MARTIAN_3DPROC_EROSION,
	MARTIAN_PROC_PCA = 120,
};

unsigned char* m_InputBuffer = new unsigned char[256 * 256 * 256];
unsigned char* m_OutputBuffer = new unsigned char[256 * 256 * 256 * 2];

namespace MartianAIUnitTest
{
	static MartianAIControl* martianAIControl;

	TEST_CLASS(MartianAIUnitTest)
	{
	public:

		TEST_CLASS_INITIALIZE(InitMartianAIUnitTest)
		{
			if (martianAIControl)
			{
				delete martianAIControl;
				martianAIControl = NULL;
			}
			system("wmic process where name='MartianAI.exe' delete");

			martianAIControl = new MartianAIControl();

			std::string mpath = "./";

			std::wstring widestr = std::wstring(mpath.begin(), mpath.end());

			WCHAR modelPath[1024] = { 0, };

			wmemcpy(modelPath, widestr.c_str(), widestr.size());

			int resultValue = martianAIControl->Init(0, 0, modelPath);
			if (resultValue != 0)
			{
				Assert::Fail(L"Create martianAIControl Failed.\n");
			}

		}
		TEST_CLASS_CLEANUP(CleanupMartianAIUnitTest)
		{
			if (martianAIControl)
			{
				delete martianAIControl;
				martianAIControl = NULL;
			}
			system("wmic process where name='MartianAI.exe' delete");
		}

		TEST_METHOD(FirstFetusSegmentationLoadCheckTest)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_SEG_FETUS_1ST);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to FirstFetusSegmentationLoadCheckTest.\n");
			}
		}
		TEST_METHOD(SecondFetusSegmentationLoadCheckTest)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_SEG_FETUS_2ND);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to SecondFetusSegmentationLoadCheckTest.\n");
			}
		}
		TEST_METHOD(CNSSegmentationLoadCheckTest)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_CNS_SEG);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to CNSSegmentationLoadCheckTest.\n");
			}
		}
		TEST_METHOD(CNSTCMeasureLoadCheckTest)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_CNS_MEASURETC);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to CNSTCMeasureLoadCheckTest.\n");
			}
		}
		TEST_METHOD(CNSTTMeasureLoadCheckTest)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_CNS_MEASURETT);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to CNSTTMeasureLoadCheckTest.\n");
			}
		}
		TEST_METHOD(CNSTVMeasureLoadCheckTest)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_CNS_MEASURETV);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to CNSTVMeasureLoadCheckTest.\n");
			}
		}
		TEST_METHOD(StyleTransferAutoInpaintLoadCheckTest)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_STYLE_TRANSFER_AUTORESTORE);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to StyleTransferAutoInpaintLoadCheckTest.\n");
			}
		}
		TEST_METHOD(StyleTransferManualInpaintLoadCheckTest)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_STYLE_TRANSFER_INPAINTING);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to StyleTransferManualInpaintLoadCheckTestt.\n");
			}
		}
		TEST_METHOD(StyleTransferEnhancementLoadCheckTest)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_STYLE_TRANSFER_ENHANCEMENT);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to StyleTransferEnhancementLoadCheckTest.\n");
			}
		}
		TEST_METHOD(StyleTransferClassificationLoadCheckTest)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_STYLE_TRANSFER_FACECLASSIFICATION);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to StyleTransferClassificationLoadCheckTest.\n");
			}
		}
		TEST_METHOD(PelvicAssis3DSegmentationLoadCheckTest)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_SEG_PELVICASSIST);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to PelvicAssis3DSegmentationLoadCheckTest.\n");
			}
		}
		TEST_METHOD(PelvicAssisMeasure1LoadCheckTest)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_SEG_PELVICMEASURE1);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to PelvicAssisMeasure1LoadCheckTest.\n");
			}
		}

		TEST_METHOD(FirstFetusSegmentationTest)
		{
			std::ifstream fin("SegmentationUnitTestData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in FirstFetusSegmentationTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 256*256*256 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_SEG_FETUS_1ST;
			inferenceInfo.xSize = 256;
			inferenceInfo.ySize = 256;
			inferenceInfo.zSize = 256;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.labelNum = 6;

			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 3;
			inferenceInfo.processOnTable[2] = 3;

			for (int i = 0; i < 16; i++)
			{
				if (i == 0)
				{
					inferenceInfo.thresholdTable[i] = 0.0;
				}
				else
				{
					inferenceInfo.thresholdTable[i] = 0.5;
				}
			}

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run FirstFetusSegmentationTest.\n");
			}
		}

		TEST_METHOD(FirstFetusSegmentationTestN)
		{
			std::ifstream fin("SegmentationUnitTestData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in FirstFetusSegmentationTestN.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 256 * 256 * 256 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_SEG_FETUS_1ST;
			inferenceInfo.xSize = 256;
			inferenceInfo.ySize = 256;
			inferenceInfo.zSize = 256;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.labelNum = 0;

			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;

			for (int i = 0; i < 16; i++)
			{
				if (i == 0)
				{
					inferenceInfo.thresholdTable[i] = 0.0;
				}
				else
				{
					inferenceInfo.thresholdTable[i] = 0.5;
				}
			}

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue == 0)
			{
				Assert::Fail(L"Failed to Run FirstFetusSegmentationTestN.\n");
			}
		}
		TEST_METHOD(SecondFetusSegmentationTest)
		{
			std::ifstream fin("SegmentationUnitTestData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in SecondFetusSegmentationTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 256 * 256 * 256 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_SEG_FETUS_2ND;
			inferenceInfo.xSize = 256;
			inferenceInfo.ySize = 256;
			inferenceInfo.zSize = 256;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.labelNum = 8;

			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;

			for (int i = 0; i < 16; i++)
			{
				if (i == 0)
				{
					inferenceInfo.thresholdTable[i] = 0.0;
				}
				else
				{
					inferenceInfo.thresholdTable[i] = 0.5;
				}
			}

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run SecondFetusSegmentationTest.\n");
			}
		}

		TEST_METHOD(SecondFetusSegmentationTestN)
		{
			std::ifstream fin("SegmentationUnitTestData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in SecondFetusSegmentationTestN.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 256 * 256 * 256 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_SEG_FETUS_2ND;
			inferenceInfo.xSize = 256;
			inferenceInfo.ySize = 256;
			inferenceInfo.zSize = 256;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.labelNum = 0;

			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;

			for (int i = 0; i < 16; i++)
			{
				if (i == 0)
				{
					inferenceInfo.thresholdTable[i] = 0.0;
				}
				else
				{
					inferenceInfo.thresholdTable[i] = 0.5;
				}
			}

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue == 0)
			{
				Assert::Fail(L"Failed to Run SecondFetusSegmentationTestN.\n");
			}
		}
		TEST_METHOD(CNSSegmentationTest)
		{
			std::ifstream fin("SegmentationUnitTestData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in CNSSegmentationTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 256 * 256 * 256 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_CNS_SEG;
			inferenceInfo.xSize = 256;
			inferenceInfo.ySize = 256;
			inferenceInfo.zSize = 256;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.labelNum = 8;

			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.thresholdTable[0] = 0.0; // 0 BG
			inferenceInfo.thresholdTable[1] = 0.5; // 1 Th
			inferenceInfo.thresholdTable[2] = 0.5; // 2 CB
			inferenceInfo.thresholdTable[3] = 0.5; // 3 CM
			inferenceInfo.thresholdTable[4] = 0.5; // 4 CP
			inferenceInfo.thresholdTable[5] = 0.5; // 5 PVC
			inferenceInfo.thresholdTable[6] = 0.5; // 6 CSP
			inferenceInfo.thresholdTable[7] = 0.5; // 7 MidLine

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);
			if (resultValue == 0)
			{
				if (inferenceInfo.errorCode != 300)
				{
					Assert::Fail(L"Failed to errorcheck CNSSegmentationTest.\n");
				}
			}
			else
			{
				Assert::Fail(L"Failed to Run CNSSegmentationTest.\n");
			}
		}

		TEST_METHOD(CNSSegmentationTestN)
		{
			memset(m_InputBuffer, 0x00, sizeof(unsigned char) * 256 * 256 * 256);

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_CNS_SEG;
			inferenceInfo.xSize = 256;
			inferenceInfo.ySize = 256;
			inferenceInfo.zSize = 256;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.labelNum = 8;

			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.thresholdTable[0] = 0.0; // 0 BG
			inferenceInfo.thresholdTable[1] = 0.5; // 1 Th
			inferenceInfo.thresholdTable[2] = 0.5; // 2 CB
			inferenceInfo.thresholdTable[3] = 0.5; // 3 CM
			inferenceInfo.thresholdTable[4] = 0.5; // 4 CP
			inferenceInfo.thresholdTable[5] = 0.5; // 5 PVC
			inferenceInfo.thresholdTable[6] = 0.5; // 6 CSP
			inferenceInfo.thresholdTable[7] = 0.5; // 7 MidLine

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);
			if (resultValue == 0)
			{
				if (inferenceInfo.errorCode == 300)
				{
					Assert::Fail(L"Failed to errorcheck CNSSegmentationTestN.\n");
				}
			}
			else
			{
				Assert::Fail(L"Failed to Run CNSSegmentationTestN.\n");
			}
		}
		TEST_METHOD(CNSTCMeasureTest)
		{
			std::ifstream fin("CNSTCMeasureData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in CNSTCMeasureTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 512 * 512 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_CNS_MEASURETC;
			inferenceInfo.xSize = 512;
			inferenceInfo.ySize = 512;
			inferenceInfo.zSize = 1;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.inferenceTime = 0;
			inferenceInfo.imageRotate = 0;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run CNSTCMeasureTest.\n");
			}
		}

		TEST_METHOD(CNSTTMeasureTest)
		{
			std::ifstream fin("CNSTTMeasureData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in CNSTTMeasureTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 512 * 512 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_CNS_MEASURETT;
			inferenceInfo.xSize = 512;
			inferenceInfo.ySize = 512;
			inferenceInfo.zSize = 1;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.inferenceTime = 0;
			inferenceInfo.imageRotate = 0;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run CNSTTMeasureTest.\n");
			}
		}

		TEST_METHOD(CNSTVMeasureTest)
		{
			std::ifstream fin("CNSTVMeasureData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in CNSTVMeasureTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 512 * 512 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_CNS_MEASURETV;
			inferenceInfo.xSize = 512;
			inferenceInfo.ySize = 512;
			inferenceInfo.zSize = 1;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.inferenceTime = 0;
			inferenceInfo.imageRotate = 0;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run CNSTVMeasureTest.\n");
			}
		}

		TEST_METHOD(StyleTransferAutoInpaintTest)
		{
			std::ifstream fin("StyleTransferData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in StyleTransferAutoInpaintTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 400 * 400 * 3 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_STYLE_TRANSFER_AUTORESTORE;
			inferenceInfo.xSize = 400;
			inferenceInfo.ySize = 400;
			inferenceInfo.zSize = 3;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.inferenceTime = 0;
			inferenceInfo.imageRotate = 0;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run StyleTransferAutoInpaintTest.\n");
			}
		}

		TEST_METHOD(StyleTransferManualInpaintTest)
		{
			std::ifstream fin("StyleTransferData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in StyleTransferManualInpaintTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 400 * 400 * 3 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_STYLE_TRANSFER_INPAINTING;
			inferenceInfo.xSize = 400;
			inferenceInfo.ySize = 400;
			inferenceInfo.zSize = 3;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.inferenceTime = 0;
			inferenceInfo.imageRotate = 0;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run StyleTransferManualInpaintTest.\n");
			}
		}

		TEST_METHOD(StyleTransferEnhancementTest)
		{
			std::ifstream fin("StyleTransferData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in StyleTransferEnhancementTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 400 * 400 * 3 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_STYLE_TRANSFER_ENHANCEMENT;
			inferenceInfo.xSize = 400;
			inferenceInfo.ySize = 400;
			inferenceInfo.zSize = 3;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.inferenceTime = 0;
			inferenceInfo.imageRotate = 0;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run StyleTransferEnhancementTest.\n");
			}
		}

		TEST_METHOD(StyleTransferClassificationTest)
		{
			std::ifstream fin("StyleTransferData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in StyleTransferEnhancementTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 400 * 400 * 3 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_STYLE_TRANSFER_FACECLASSIFICATION;
			inferenceInfo.xSize = 400;
			inferenceInfo.ySize = 400;
			inferenceInfo.zSize = 3;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.inferenceTime = 0;
			inferenceInfo.imageRotate = 0;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run StyleTransferEnhancementTest.\n");
			}
		}

		TEST_METHOD(gaussian3DFilterKernelSize3Test)
		{
			std::ifstream fin("SegmentationUnitTestData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in gaussian3DFilterKernelSize3Test.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 256 * 256 * 256 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_3DPROC_GAUSSION;
			inferenceInfo.xSize = 256;
			inferenceInfo.ySize = 256;
			inferenceInfo.zSize = 256;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.labelNum = 7;
			inferenceInfo.kernelSize = 3;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run gaussian3DFilterKernelSize3Test.\n");
			}
		}

		TEST_METHOD(gaussian3DFilterKernelSize5Test)
		{
			std::ifstream fin("SegmentationUnitTestData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in gaussian3DFilterKernelSize5Test.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 256 * 256 * 256 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_3DPROC_GAUSSION;
			inferenceInfo.xSize = 256;
			inferenceInfo.ySize = 256;
			inferenceInfo.zSize = 256;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.labelNum = 7;
			inferenceInfo.kernelSize = 5;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run gaussian3DFilterKernelSize5Test.\n");
			}
		}
		TEST_METHOD(PelvicAssist3DSegmentationTest)
		{
			std::ifstream fin("PelvicAssist.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in PelvicAssist3DSegmentationTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 256 * 256 * 256 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_SEG_PELVICASSIST;
			inferenceInfo.xSize = 256;
			inferenceInfo.ySize = 256;
			inferenceInfo.zSize = 256;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.labelNum = 6;

			//Gaussian Smooth Filter On - 1, Off - 0
			inferenceInfo.gaussianFilterMode = 0;

			// Dilation - 1, Erosion - 2, Both - 3
			inferenceInfo.morphologyMode = 0;

			// ProcessOnTable
			inferenceInfo.processOnTable[0] = 0;        // BG 
			inferenceInfo.processOnTable[1] = 1;        // ANUS
			inferenceInfo.processOnTable[2] = 1;        // VARGINA
			inferenceInfo.processOnTable[3] = 1;        // URETHA
			inferenceInfo.processOnTable[4] = 1;        // PELVICBRIM
			inferenceInfo.processOnTable[5] = 1;        // BLADDER

			inferenceInfo.thresholdTable[0] = 0.0;      // BG
			inferenceInfo.thresholdTable[1] = 0.5;      // ANUS
			inferenceInfo.thresholdTable[2] = 0.5;      // VARGINA
			inferenceInfo.thresholdTable[3] = 0.5;      // URETHA
			inferenceInfo.thresholdTable[4] = 0.5;      // PELVICBRIM
			inferenceInfo.thresholdTable[5] = 0.5;      // BLADDER

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run PelvicAssist3DSegmentationTest.\n");
			}
		}

		TEST_METHOD(PelvicAssistMeasureTest)
		{
			std::ifstream fin("PelvicMeasureConcatData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in PelvicAssistMeasureTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 400 * 400 * 4 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_SEG_PELVICMEASURE1;
			inferenceInfo.xSize = 400;
			inferenceInfo.ySize = 400;
			inferenceInfo.zSize = 4;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.inferenceTime = 0;
			inferenceInfo.imageRotate = 0;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run PelvicAssistMeasureTest.\n");
			}
		}

		TEST_METHOD(AI5DNT3DSegmentation)
		{
			std::ifstream fin("NT3Data.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in AI5DNT3DSegmentation.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 256 * 256 * 256 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_SEG_NT;
			inferenceInfo.xSize = 256;
			inferenceInfo.ySize = 256;
			inferenceInfo.zSize = 256;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.inferenceTime = 0;
			inferenceInfo.imageRotate = 0;	
			inferenceInfo.labelNum = 7;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run AI5DNT3DSegmentation.\n");
			}
		}

		TEST_METHOD(AI5DNT2DSegmentation)
		{
			std::ifstream fin("CNSTCMeasureData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in AI5DNT2DSegmentation.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 512 * 512 * 1 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_SEG_NTMEASURE;
			inferenceInfo.xSize = 512;
			inferenceInfo.ySize = 512;
			inferenceInfo.zSize = 1;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.inferenceTime = 0;
			inferenceInfo.imageRotate = 0;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run AI5DNT2DSegmentation.\n");
			}
		}
	};

	TEST_CLASS(MartianAIUnitTestCPU)
	{
	public:

		TEST_CLASS_INITIALIZE(InitMartianAIUnitTestCPU)
		{
			if (martianAIControl)
			{
				delete martianAIControl;
				martianAIControl = NULL;
			}
			system("wmic process where name='MartianAI.exe' delete");

			martianAIControl = new MartianAIControl();

			std::string mpath = "./";

			std::wstring widestr = std::wstring(mpath.begin(), mpath.end());

			WCHAR modelPath[1024] = { 0, };

			wmemcpy(modelPath, widestr.c_str(), widestr.size());

			int resultValue = martianAIControl->Init(1, 0, modelPath);
			if (resultValue != 0)
			{
				Assert::Fail(L"Create martianAIControlCPU Failed.\n");
			}

		}
		TEST_CLASS_CLEANUP(CleanupMartianAIUnitTestCPU)
		{
			if (martianAIControl)
			{
				delete martianAIControl;
				martianAIControl = NULL;
			}
			system("wmic process where name='MartianAI.exe' delete");
		}

		TEST_METHOD(FirstFetusSegmentationLoadCheckTestCPU)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_SEG_FETUS_1ST);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to FirstFetusSegmentationLoadCheckTest.\n");
			}
		}
		TEST_METHOD(SecondFetusSegmentationLoadCheckTestCPU)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_SEG_FETUS_2ND);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to SecondFetusSegmentationLoadCheckTest.\n");
			}
		}
		TEST_METHOD(CNSSegmentationLoadCheckTestCPU)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_CNS_SEG);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to CNSSegmentationLoadCheckTest.\n");
			}
		}
		TEST_METHOD(CNSTCMeasureLoadCheckTestCPU)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_CNS_MEASURETC);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to CNSTCMeasureLoadCheckTest.\n");
			}
		}
		TEST_METHOD(CNSTTMeasureLoadCheckTestCPU)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_CNS_MEASURETT);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to CNSTTMeasureLoadCheckTest.\n");
			}
		}
		TEST_METHOD(CNSTVMeasureLoadCheckTestCPU)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_CNS_MEASURETV);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to CNSTVMeasureLoadCheckTest.\n");
			}
		}
		TEST_METHOD(StyleTransferAutoInpaintLoadCheckTestCPU)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_STYLE_TRANSFER_AUTORESTORE);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to StyleTransferAutoInpaintLoadCheckTest.\n");
			}
		}
		TEST_METHOD(StyleTransferManualInpaintLoadCheckTestCPU)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_STYLE_TRANSFER_INPAINTING);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to StyleTransferManualInpaintLoadCheckTestt.\n");
			}
		}
		TEST_METHOD(StyleTransferEnhancementLoadCheckTestCPU)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_STYLE_TRANSFER_ENHANCEMENT);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to StyleTransferEnhancementLoadCheckTest.\n");
			}
		}
		TEST_METHOD(StyleTransferClassificationLoadCheckTestCPU)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_STYLE_TRANSFER_FACECLASSIFICATION);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to StyleTransferClassificationLoadCheckTest.\n");
			}
		}
		TEST_METHOD(PelvicAssis3DSegmentationLoadCheckTestCPU)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_SEG_PELVICASSIST);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to PelvicAssis3DSegmentationLoadCheckTest.\n");
			}
		}
		TEST_METHOD(PelvicAssisMeasure1LoadCheckTestCPU)
		{
			int modelLoadDone = martianAIControl->CheckInitDone(MARTIAN_SEG_PELVICMEASURE1);
			if (modelLoadDone != 0)
			{
				Assert::Fail(L"Failed to PelvicAssisMeasure1LoadCheckTest.\n");
			}
		}

		TEST_METHOD(FirstFetusSegmentationTestCPU)
		{
			std::ifstream fin("SegmentationUnitTestData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in FirstFetusSegmentationTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 256 * 256 * 256 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_SEG_FETUS_1ST;
			inferenceInfo.xSize = 256;
			inferenceInfo.ySize = 256;
			inferenceInfo.zSize = 256;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.labelNum = 6;

			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 3;
			inferenceInfo.processOnTable[2] = 3;

			for (int i = 0; i < 16; i++)
			{
				if (i == 0)
				{
					inferenceInfo.thresholdTable[i] = 0.0;
				}
				else
				{
					inferenceInfo.thresholdTable[i] = 0.5;
				}
			}

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run FirstFetusSegmentationTest.\n");
			}
		}

		TEST_METHOD(FirstFetusSegmentationTestNCPU)
		{
			std::ifstream fin("SegmentationUnitTestData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in FirstFetusSegmentationTestN.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 256 * 256 * 256 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_SEG_FETUS_1ST;
			inferenceInfo.xSize = 256;
			inferenceInfo.ySize = 256;
			inferenceInfo.zSize = 256;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.labelNum = 0;

			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;

			for (int i = 0; i < 16; i++)
			{
				if (i == 0)
				{
					inferenceInfo.thresholdTable[i] = 0.0;
				}
				else
				{
					inferenceInfo.thresholdTable[i] = 0.5;
				}
			}

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue == 0)
			{
				Assert::Fail(L"Failed to Run FirstFetusSegmentationTestN.\n");
			}
		}
		TEST_METHOD(SecondFetusSegmentationTestCPU)
		{
			std::ifstream fin("SegmentationUnitTestData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in SecondFetusSegmentationTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 256 * 256 * 256 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_SEG_FETUS_2ND;
			inferenceInfo.xSize = 256;
			inferenceInfo.ySize = 256;
			inferenceInfo.zSize = 256;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.labelNum = 8;

			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;

			for (int i = 0; i < 16; i++)
			{
				if (i == 0)
				{
					inferenceInfo.thresholdTable[i] = 0.0;
				}
				else
				{
					inferenceInfo.thresholdTable[i] = 0.5;
				}
			}

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run SecondFetusSegmentationTest.\n");
			}
		}

		TEST_METHOD(SecondFetusSegmentationTestNCPU)
		{
			std::ifstream fin("SegmentationUnitTestData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in SecondFetusSegmentationTestN.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 256 * 256 * 256 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_SEG_FETUS_2ND;
			inferenceInfo.xSize = 256;
			inferenceInfo.ySize = 256;
			inferenceInfo.zSize = 256;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.labelNum = 0;

			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;

			for (int i = 0; i < 16; i++)
			{
				if (i == 0)
				{
					inferenceInfo.thresholdTable[i] = 0.0;
				}
				else
				{
					inferenceInfo.thresholdTable[i] = 0.5;
				}
			}

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue == 0)
			{
				Assert::Fail(L"Failed to Run SecondFetusSegmentationTestN.\n");
			}
		}
		TEST_METHOD(CNSSegmentationTestCPU)
		{
			std::ifstream fin("SegmentationUnitTestData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in CNSSegmentationTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 256 * 256 * 256 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_CNS_SEG;
			inferenceInfo.xSize = 256;
			inferenceInfo.ySize = 256;
			inferenceInfo.zSize = 256;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.labelNum = 8;

			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.thresholdTable[0] = 0.0; // 0 BG
			inferenceInfo.thresholdTable[1] = 0.5; // 1 Th
			inferenceInfo.thresholdTable[2] = 0.5; // 2 CB
			inferenceInfo.thresholdTable[3] = 0.5; // 3 CM
			inferenceInfo.thresholdTable[4] = 0.5; // 4 CP
			inferenceInfo.thresholdTable[5] = 0.5; // 5 PVC
			inferenceInfo.thresholdTable[6] = 0.5; // 6 CSP
			inferenceInfo.thresholdTable[7] = 0.5; // 7 MidLine

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);
			if (resultValue == 0)
			{
				if (inferenceInfo.errorCode != 300)
				{
					Assert::Fail(L"Failed to errorcheck CNSSegmentationTest.\n");
				}
			}
			else
			{
				Assert::Fail(L"Failed to Run CNSSegmentationTest.\n");
			}
		}

		TEST_METHOD(CNSSegmentationTestNCPU)
		{
			memset(m_InputBuffer, 0x00, sizeof(unsigned char) * 256 * 256 * 256);

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_CNS_SEG;
			inferenceInfo.xSize = 256;
			inferenceInfo.ySize = 256;
			inferenceInfo.zSize = 256;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.labelNum = 8;

			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.thresholdTable[0] = 0.0; // 0 BG
			inferenceInfo.thresholdTable[1] = 0.5; // 1 Th
			inferenceInfo.thresholdTable[2] = 0.5; // 2 CB
			inferenceInfo.thresholdTable[3] = 0.5; // 3 CM
			inferenceInfo.thresholdTable[4] = 0.5; // 4 CP
			inferenceInfo.thresholdTable[5] = 0.5; // 5 PVC
			inferenceInfo.thresholdTable[6] = 0.5; // 6 CSP
			inferenceInfo.thresholdTable[7] = 0.5; // 7 MidLine

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);
			if (resultValue == 0)
			{
				if (inferenceInfo.errorCode == 300)
				{
					Assert::Fail(L"Failed to errorcheck CNSSegmentationTestN.\n");
				}
			}
			else
			{
				Assert::Fail(L"Failed to Run CNSSegmentationTestN.\n");
			}
		}
		TEST_METHOD(CNSTCMeasureTestCPU)
		{
			std::ifstream fin("CNSTCMeasureData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in CNSTCMeasureTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 512 * 512 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_CNS_MEASURETC;
			inferenceInfo.xSize = 512;
			inferenceInfo.ySize = 512;
			inferenceInfo.zSize = 1;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.inferenceTime = 0;
			inferenceInfo.imageRotate = 0;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run CNSTCMeasureTest.\n");
			}
		}

		TEST_METHOD(CNSTTMeasureTestCPU)
		{
			std::ifstream fin("CNSTTMeasureData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in CNSTTMeasureTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 512 * 512 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_CNS_MEASURETT;
			inferenceInfo.xSize = 512;
			inferenceInfo.ySize = 512;
			inferenceInfo.zSize = 1;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.inferenceTime = 0;
			inferenceInfo.imageRotate = 0;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run CNSTTMeasureTest.\n");
			}
		}

		TEST_METHOD(CNSTVMeasureTestCPU)
		{
			std::ifstream fin("CNSTVMeasureData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in CNSTVMeasureTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 512 * 512 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_CNS_MEASURETV;
			inferenceInfo.xSize = 512;
			inferenceInfo.ySize = 512;
			inferenceInfo.zSize = 1;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.inferenceTime = 0;
			inferenceInfo.imageRotate = 0;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run CNSTVMeasureTest.\n");
			}
		}

		TEST_METHOD(StyleTransferAutoInpaintTestCPU)
		{
			std::ifstream fin("StyleTransferData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in StyleTransferAutoInpaintTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 400 * 400 * 3 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_STYLE_TRANSFER_AUTORESTORE;
			inferenceInfo.xSize = 400;
			inferenceInfo.ySize = 400;
			inferenceInfo.zSize = 3;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.inferenceTime = 0;
			inferenceInfo.imageRotate = 0;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run StyleTransferAutoInpaintTest.\n");
			}
		}

		TEST_METHOD(StyleTransferManualInpaintTestCPU)
		{
			std::ifstream fin("StyleTransferData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in StyleTransferManualInpaintTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 400 * 400 * 3 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_STYLE_TRANSFER_INPAINTING;
			inferenceInfo.xSize = 400;
			inferenceInfo.ySize = 400;
			inferenceInfo.zSize = 3;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.inferenceTime = 0;
			inferenceInfo.imageRotate = 0;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run StyleTransferManualInpaintTest.\n");
			}
		}

		TEST_METHOD(StyleTransferEnhancementTestCPU)
		{
			std::ifstream fin("StyleTransferData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in StyleTransferEnhancementTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 400 * 400 * 3 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_STYLE_TRANSFER_ENHANCEMENT;
			inferenceInfo.xSize = 400;
			inferenceInfo.ySize = 400;
			inferenceInfo.zSize = 3;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.inferenceTime = 0;
			inferenceInfo.imageRotate = 0;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run StyleTransferEnhancementTest.\n");
			}
		}

		TEST_METHOD(StyleTransferClassificationTestCPU)
		{
			std::ifstream fin("StyleTransferData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in StyleTransferEnhancementTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 400 * 400 * 3 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_STYLE_TRANSFER_FACECLASSIFICATION;
			inferenceInfo.xSize = 400;
			inferenceInfo.ySize = 400;
			inferenceInfo.zSize = 3;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.inferenceTime = 0;
			inferenceInfo.imageRotate = 0;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run StyleTransferEnhancementTest.\n");
			}
		}

		TEST_METHOD(gaussian3DFilterKernelSize3TestCPU)
		{
			std::ifstream fin("SegmentationUnitTestData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in gaussian3DFilterKernelSize3Test.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 256 * 256 * 256 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_3DPROC_GAUSSION;
			inferenceInfo.xSize = 256;
			inferenceInfo.ySize = 256;
			inferenceInfo.zSize = 256;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.labelNum = 7;
			inferenceInfo.kernelSize = 3;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run gaussian3DFilterKernelSize3Test.\n");
			}
		}

		TEST_METHOD(gaussian3DFilterKernelSize5TestCPU)
		{
			std::ifstream fin("SegmentationUnitTestData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in gaussian3DFilterKernelSize5Test.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 256 * 256 * 256 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_3DPROC_GAUSSION;
			inferenceInfo.xSize = 256;
			inferenceInfo.ySize = 256;
			inferenceInfo.zSize = 256;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.labelNum = 7;
			inferenceInfo.kernelSize = 5;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run gaussian3DFilterKernelSize5Test.\n");
			}
		}
		TEST_METHOD(PelvicAssist3DSegmentationTestCPU)
		{
			std::ifstream fin("PelvicAssist.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in PelvicAssist3DSegmentationTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 256 * 256 * 256 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_SEG_PELVICASSIST;
			inferenceInfo.xSize = 256;
			inferenceInfo.ySize = 256;
			inferenceInfo.zSize = 256;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.labelNum = 6;

			//Gaussian Smooth Filter On - 1, Off - 0
			inferenceInfo.gaussianFilterMode = 0;

			// Dilation - 1, Erosion - 2, Both - 3
			inferenceInfo.morphologyMode = 0;

			// ProcessOnTable
			inferenceInfo.processOnTable[0] = 0;        // BG 
			inferenceInfo.processOnTable[1] = 1;        // ANUS
			inferenceInfo.processOnTable[2] = 1;        // VARGINA
			inferenceInfo.processOnTable[3] = 1;        // URETHA
			inferenceInfo.processOnTable[4] = 1;        // PELVICBRIM
			inferenceInfo.processOnTable[5] = 1;        // BLADDER

			inferenceInfo.thresholdTable[0] = 0.0;      // BG
			inferenceInfo.thresholdTable[1] = 0.5;      // ANUS
			inferenceInfo.thresholdTable[2] = 0.5;      // VARGINA
			inferenceInfo.thresholdTable[3] = 0.5;      // URETHA
			inferenceInfo.thresholdTable[4] = 0.5;      // PELVICBRIM
			inferenceInfo.thresholdTable[5] = 0.5;      // BLADDER

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run PelvicAssist3DSegmentationTest.\n");
			}
		}

		TEST_METHOD(PelvicAssistMeasureTestCPU)
		{
			std::ifstream fin("PelvicMeasureConcatData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in PelvicAssistMeasureTest.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 400 * 400 * 4 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_SEG_PELVICMEASURE1;
			inferenceInfo.xSize = 400;
			inferenceInfo.ySize = 400;
			inferenceInfo.zSize = 4;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.inferenceTime = 0;
			inferenceInfo.imageRotate = 0;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run PelvicAssistMeasureTest.\n");
			}
		}

		TEST_METHOD(AI5DNT3DSegmentationCPU)
		{
			std::ifstream fin("NT3Data.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in AI5DNT3DSegmentation.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 256 * 256 * 256 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_SEG_NT;
			inferenceInfo.xSize = 256;
			inferenceInfo.ySize = 256;
			inferenceInfo.zSize = 256;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.inferenceTime = 0;
			inferenceInfo.imageRotate = 0;
			inferenceInfo.labelNum = 7;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run AI5DNT3DSegmentation.\n");
			}
		}

		TEST_METHOD(AI5DNT2DSegmentationCPU)
		{
			std::ifstream fin("CNSTCMeasureData.bin", std::ios::binary);
			if (!fin)
			{
				Assert::Fail(L"Failed to FileRead in AI5DNT2DSegmentation.\n");
			}
			fin.read(reinterpret_cast<char*>(m_InputBuffer), 512 * 512 * 1 * sizeof(unsigned char));

			InferenceInfo inferenceInfo;
			inferenceInfo.InferenceMode = MARTIAN_SEG_NTMEASURE;
			inferenceInfo.xSize = 512;
			inferenceInfo.ySize = 512;
			inferenceInfo.zSize = 1;
			memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
			memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
			inferenceInfo.gaussianFilterMode = 0;
			inferenceInfo.morphologyMode = 0;
			inferenceInfo.inferenceTime = 0;
			inferenceInfo.imageRotate = 0;

			int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

			if (resultValue != 0)
			{
				Assert::Fail(L"Failed to Run AI5DNT2DSegmentation.\n");
			}
		}
	};
}
