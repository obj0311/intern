
// MartianAIHostDlg.cpp: 구현 파일
//

#include "pch.h"
#include "framework.h"
#include "MartianAIHost.h"
#include "MartianAIHostDlg.h"
#include "afxdialogex.h"
#include <chrono>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

using namespace std;
// 응용 프로그램 정보에 사용되는 CAboutDlg 대화 상자입니다.

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

// 구현입니다.
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CMartianAIHostDlg 대화 상자

CMartianAIHostDlg::CMartianAIHostDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_MARTIANAIHOST_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);

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

	if (m_InputImageBuffer == NULL)
	{
		m_InputImageBuffer = new unsigned char[1000 * 1000 * 3];
		memset(m_InputImageBuffer, 0x00, sizeof(unsigned char) * 1000 * 1000 * 3);
	}

	if (m_OutputImageBuffer == NULL)
	{
		m_OutputImageBuffer = new unsigned char[1000 * 1000 * 3];
		memset(m_OutputImageBuffer, 0x00, sizeof(unsigned char) * 1000 * 1000 * 3);
	}
	//GetCurrentBasePath();
}

void CMartianAIHostDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_PIC, m_picture_control);
	DDX_Control(pDX, IDC_SLIDER1, m_Slider1);
	DDX_Control(pDX, IDC_EDIT_SLIDER1, m_Edit_Slider1);
	DDX_Control(pDX, IDC_PIC1, m_picture1_control);
	DDX_Control(pDX, IDC_PIC2, m_picture2_control);
	DDX_Control(pDX, IDC_PIC3, m_picture3_control);
	DDX_Control(pDX, IDC_SLIDER2, m_Slider2);
	DDX_Control(pDX, IDC_SLIDER3, m_Slider3);
	DDX_Control(pDX, IDC_EDIT_SLIDER2, m_Edit_Slider2);
	DDX_Control(pDX, IDC_EDIT_SLIDER3, m_Edit_Slider3);
}

BEGIN_MESSAGE_MAP(CMartianAIHostDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON_INIT, &CMartianAIHostDlg::OnBnClickedButtonInit)
	ON_BN_CLICKED(IDC_BUTTON_MODE1, &CMartianAIHostDlg::OnBnClickedButtonMode1)
	ON_BN_CLICKED(IDC_BUTTON_MODE2, &CMartianAIHostDlg::OnBnClickedButtonMode2)
	ON_BN_CLICKED(IDC_BUTTON_MODE3, &CMartianAIHostDlg::OnBnClickedButtonMode3)
	ON_BN_CLICKED(IDC_BUTTON_MODE4, &CMartianAIHostDlg::OnBnClickedButtonMode4)
	ON_WM_DESTROY()
	ON_BN_CLICKED(IDC_BUTTON_GAUSSIAN, &CMartianAIHostDlg::OnBnClickedButtonGaussian)
	ON_BN_CLICKED(IDC_BUTTON_DILATION, &CMartianAIHostDlg::OnBnClickedButtonDilation)
	ON_BN_CLICKED(IDC_BUTTON_EROSION, &CMartianAIHostDlg::OnBnClickedButtonErosion)
	ON_BN_CLICKED(IDC_BUTTON_PCA, &CMartianAIHostDlg::OnBnClickedButtonPca)
	ON_WM_HSCROLL()
	ON_BN_CLICKED(IDC_BUTTON_MODE5, &CMartianAIHostDlg::OnBnClickedButtonMode5)
	ON_BN_CLICKED(IDC_BUTTON_MODE6, &CMartianAIHostDlg::OnBnClickedButtonMode6)
	ON_BN_CLICKED(IDC_BUTTON_MODE7, &CMartianAIHostDlg::OnBnClickedButtonMode7)
	ON_BN_CLICKED(IDC_BUTTON_MODENT, &CMartianAIHostDlg::OnBnClickedButtonModent)
	ON_BN_CLICKED(IDC_BUTTON_ENCRYPTION, &CMartianAIHostDlg::OnBnClickedButtonEncryption)
	ON_BN_CLICKED(IDC_BUTTON_FACECLASSIFICATION, &CMartianAIHostDlg::OnBnClickedButtonFaceclassification)
	ON_BN_CLICKED(IDC_BUTTON_MODE_PELVICASSIST, &CMartianAIHostDlg::OnBnClickedButtonModePelvicAssist)
	ON_BN_CLICKED(IDC_BUTTON_MODE_PELVICMEASURE, &CMartianAIHostDlg::OnBnClickedButtonModePelvicmeasure)
	ON_BN_CLICKED(IDC_BUTTON_MODENTMEASURE, &CMartianAIHostDlg::OnBnClickedButtonModentmeasure)
END_MESSAGE_MAP()


// CMartianAIHostDlg 메시지 처리기

BOOL CMartianAIHostDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();
	CreateDirectory(_T("./OutData"), NULL);

	// 시스템 메뉴에 "정보..." 메뉴 항목을 추가합니다.

	// IDM_ABOUTBOX는 시스템 명령 범위에 있어야 합니다.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 이 대화 상자의 아이콘을 설정합니다.  응용 프로그램의 주 창이 대화 상자가 아닐 경우에는
	//  프레임워크가 이 작업을 자동으로 수행합니다.
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정합니다.
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정합니다.

	// TODO: 여기에 추가 초기화 작업을 추가합니다.

	SetDlgItemText(IDC_EDIT_FILE1, L"SegResnetData_1.bin");
	SetDlgItemText(IDC_EDIT_FILE2, L"SegResnetData_2.bin");
	SetDlgItemText(IDC_EDIT_FILE3, L"SegResnetData_3.bin");
	SetDlgItemText(IDC_EDIT_FILE4, L"StyleTransferData.bmp");
	SetDlgItemText(IDC_EDIT_FILE5, L"CNSMeasre1Data.bin");
	SetDlgItemText(IDC_EDIT_FILE6, L"CNSMeasre2Data.bin");
	SetDlgItemText(IDC_EDIT_FILE7, L"CNSMeasre3Data.bin");
	SetDlgItemText(IDC_EDIT_FILENT, L"NT3Data.bin");
	SetDlgItemText(IDC_EDIT_FILENTMEASURE, L"NTMeasureData.bmp");
	SetDlgItemText(IDC_EDIT_FILE_PELVICASSIST, L"PelvicAssist.bin");

	SetDlgItemText(IDC_EDIT_FILEPCA, L"PCAData.bin");

	SetDlgItemText(IDC_EDIT_FILEGAU, L"./OutData/outputData_3DVolume.bin");
	SetDlgItemText(IDC_EDIT_FILEDI, L"./OutData/outputData_3DVolume.bin");
	SetDlgItemText(IDC_EDIT_FILEER, L"./OutData/outputData_3DVolume.bin");

	SetDlgItemText(IDC_EDIT_KERNEL, L"3");
	SetDlgItemText(IDC_EDIT_LABELNUMDI, L"6");
	SetDlgItemText(IDC_EDIT_LABELNUMER, L"6");
	SetDlgItemText(IDC_EDIT_PROCONDI, L"3");
	SetDlgItemText(IDC_EDIT_PROCONER, L"3");

	SetDlgItemText(IDC_EDIT_GFVSIZE, L"128");
	SetDlgItemText(IDC_EDIT_DIVSIZE, L"128");
	SetDlgItemText(IDC_EDIT_ERVSIZE, L"128");

	SetDlgItemText(IDC_EDIT_SEG1SIZE, L"256");
	SetDlgItemText(IDC_EDIT_SEG2SIZE, L"256");
	SetDlgItemText(IDC_EDIT_SEG3SIZE, L"256");
	SetDlgItemText(IDC_EDIT_NTSIZE, L"256");
	SetDlgItemText(IDC_EDIT_PELVICASSIST_SIZE, L"256");

	SetDlgItemText(IDC_EDIT_CNSM1SIZE, L"512");
	SetDlgItemText(IDC_EDIT_CNSM2SIZE, L"512");
	SetDlgItemText(IDC_EDIT_CNSM3SIZE, L"512");


	SetDlgItemText(IDC_EDIT_LABELNUMPCA, L"6");
	SetDlgItemText(IDC_EDIT_PCALABEL, L"8");
	SetDlgItemText(IDC_EDIT_PCAVSIZE, L"256");
	SetDlgItemText(IDC_EDIT_NTSIZE2, L"512");

	SetDlgItemText(IDC_EDIT_FOLDERENCRYPTION, L"./AIModel/");
	SetDlgItemText(IDC_EDIT_FILE_FACECLASSIFICATION, L"FaceClassification.bmp");

	SetDlgItemText(IDC_EDIT_FILE_PELVICASSISTMEASURE, L"PelvicMeasureConcatData.bin");
	SetDlgItemText(IDC_EDIT_PELVICMEASURE_SIZE, L"400");

	m_Slider1.SetRange(0, 127);
	m_Slider1.SetRangeMin(0);
	m_Slider1.SetRangeMax(127);

	m_Slider2.SetRange(0, 127);
	m_Slider2.SetRangeMin(0);
	m_Slider2.SetRangeMax(127);

	m_Slider3.SetRange(0, 127);
	m_Slider3.SetRangeMin(0);
	m_Slider3.SetRangeMax(127);

	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

void CMartianAIHostDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 대화 상자에 최소화 단추를 추가할 경우 아이콘을 그리려면
//  아래 코드가 필요합니다.  문서/뷰 모델을 사용하는 MFC 애플리케이션의 경우에는
//  프레임워크에서 이 작업을 자동으로 수행합니다.

void CMartianAIHostDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 그리기를 위한 디바이스 컨텍스트입니다.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 클라이언트 사각형에서 아이콘을 가운데에 맞춥니다.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 아이콘을 그립니다.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서
//  이 함수를 호출합니다.
HCURSOR CMartianAIHostDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CMartianAIHostDlg::OnBnClickedButtonInit()
{
	//// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	std::chrono::steady_clock::time_point start_Init = std::chrono::steady_clock::now();

	martianAIControl = new MartianAIControl();

	int deviceMode = ((CButton*)GetDlgItem(IDC_CHECK_CPU))->GetCheck();	// 0 - GPU. 1 - CPU
	int gpuSelection = 0;
	std::string mpath = "./";
	
	std::wstring widestr = std::wstring(mpath.begin(), mpath.end());

	WCHAR modelPath[1024] = { 0, };

	wmemcpy(modelPath, widestr.c_str(), widestr.size());

	int resultValue = martianAIControl->Init(deviceMode, gpuSelection, modelPath);

	int model1LoadDone = martianAIControl->CheckInitDone(MARTIAN_SEG_FETUS_1ST);
	int model2LoadDone = martianAIControl->CheckInitDone(MARTIAN_SEG_FETUS_2ND);
	int model3LoadDone = martianAIControl->CheckInitDone(MARTIAN_STYLE_TRANSFER_INPAINTING);
	int model4LoadDone = martianAIControl->CheckInitDone(MARTIAN_STYLE_TRANSFER_ENHANCEMENT);
	int model5LoadDone = martianAIControl->CheckInitDone(MARTIAN_CNS_SEG);
	int model6LoadDone = martianAIControl->CheckInitDone(MARTIAN_CNS_MEASURETC);
	int model7LoadDone = martianAIControl->CheckInitDone(MARTIAN_CNS_MEASURETT);
	int model8LoadDone = martianAIControl->CheckInitDone(MARTIAN_CNS_MEASURETV);
	int model9LoadDone = martianAIControl->CheckInitDone(MARTIAN_STYLE_TRANSFER_AUTORESTORE);
	int model10LoadDone = martianAIControl->CheckInitDone(MARTIAN_SEG_NT);
	int model11LoadDone = martianAIControl->CheckInitDone(MARTIAN_STYLE_TRANSFER_FACECLASSIFICATION);
	int model12LoadDone = martianAIControl->CheckInitDone(MARTIAN_SEG_PELVICASSIST);
	int model13LoadDone = martianAIControl->CheckInitDone(MARTIAN_SEG_PELVICMEASURE);
	int model14LoadDone = martianAIControl->CheckInitDone(MARTIAN_SEG_NTMEASURE);

	std::chrono::steady_clock::time_point end_Init = std::chrono::steady_clock::now();
	int initTime = std::chrono::duration_cast<std::chrono::milliseconds>(end_Init - start_Init).count();

	CString ModelLoad;
	ModelLoad.Format(_T("[f1-%d, f2-%d, s1-%d, s2-%d, s3-%d, s4-%d, c1-%d, cmtc-%d, cmtt-%d, cmtv-%d, NT-%d, NTMeasure-%d ,PelvicAssist-%d, PelvicMeasure-%d] InitTime = %d"),
		model1LoadDone, model2LoadDone, model3LoadDone, model4LoadDone, model9LoadDone, model11LoadDone, model5LoadDone, model6LoadDone, model7LoadDone, model8LoadDone, model10LoadDone, model14LoadDone, model12LoadDone, model13LoadDone, initTime);

	if (resultValue == 0)
	{
		MessageBox(L"Init Done" + ModelLoad);
	}
	else
	{
		MessageBox(L"Init Failed");
	}

}


void CMartianAIHostDlg::OnBnClickedButtonMode1()
{
	CString strFile;
	GetDlgItemText(IDC_EDIT_FILE1, strFile);
	CString strInVolumeSize;
	GetDlgItemText(IDC_EDIT_SEG1SIZE, strInVolumeSize);
	int inVolumeSize = _ttoi(strInVolumeSize);

	int size = inVolumeSize * inVolumeSize * inVolumeSize;
	int outSize = 128 * 128 * 128;

	std::ifstream fin(strFile, std::ios::binary);

	if (!fin)
	{
		MessageBox(L"Check File Name");
		return;
	}

	// 한번에 Loading
	fin.read(reinterpret_cast<char*>(m_InputBuffer), size * sizeof(unsigned char));

	InferenceInfo inferenceInfo;
	inferenceInfo.InferenceMode = MARTIAN_SEG_FETUS_1ST;
	inferenceInfo.xSize = inVolumeSize;
	inferenceInfo.ySize = inVolumeSize;
	inferenceInfo.zSize = inVolumeSize;
	memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
	memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
	inferenceInfo.labelNum = 6;

	//Gaussian Smooth Filter On - 1, Off - 0
	inferenceInfo.gaussianFilterMode = 0;

	// Dilation - 1, Erosion - 2, Both - 3
	inferenceInfo.morphologyMode = 0;
	// Dilation - 1, Erosion - 2, Both - 3
	inferenceInfo.processOnTable[1] = 2;		// 1 fluid
	inferenceInfo.processOnTable[2] = 1;		// 2 fetus
	inferenceInfo.processOnTable[3] = 1;		// 3 ucord
	inferenceInfo.processOnTable[4] = 2;		// 4 uterus
	inferenceInfo.processOnTable[5] = 2;		// 5 placenta


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
		m_PlaneSize = inferenceInfo.xSize;
		m_PixelRatio = 1;

		SliderSetting(1, 0, inferenceInfo.xSize, inferenceInfo.xSize / 2);

		int plane = 64;
		CImage outImage1;
		outImage1.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);
		CImage outImage2;
		outImage2.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);
		CImage outImage3;
		outImage3.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);

		COLORREF px1;
		COLORREF px2;
		COLORREF px3;

		unsigned char RR1, GG1, BB1;
		unsigned char RR2, GG2, BB2;
		unsigned char RR3, GG3, BB3;

		for (int i = 0; i < inferenceInfo.xSize; i++)
		{
			for (int k = 0; k < inferenceInfo.xSize; k++)
			{
				RR2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);
				GG2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);
				BB2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);

				RR1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);
				GG1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);
				BB1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);

				RR3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);
				GG3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);
				BB3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);

				px1 = RGB(RR1, GG1, BB1);
				px2 = RGB(RR2, GG2, BB2);
				px3 = RGB(RR3, GG3, BB3);

				outImage1.SetPixel(i, k, px1);
				outImage2.SetPixel(i, k, px2);
				outImage3.SetPixel(i, k, px3);

			}
		}
		
		CRect rect1;
		m_picture1_control.GetWindowRect(rect1);
		CDC* dc1;
		dc1 = m_picture1_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc1->m_hDC, HALFTONE);
		outImage1.StretchBlt(dc1->m_hDC, 0, 0, rect1.Width(), rect1.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc1);//DC 해제

		CRect rect2;
		m_picture2_control.GetWindowRect(rect2);
		CDC* dc2;
		dc2 = m_picture2_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc2->m_hDC, HALFTONE);
		outImage2.StretchBlt(dc2->m_hDC, 0, 0, rect2.Width(), rect2.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc2);//DC 해제

		CRect rect3;
		m_picture3_control.GetWindowRect(rect3);
		CDC* dc3;
		dc3 = m_picture3_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc3->m_hDC, HALFTONE);
		outImage3.StretchBlt(dc3->m_hDC, 0, 0, rect3.Width(), rect3.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc3);//DC 해제

		std::ofstream FILE("./OutData/outputData_3DVolume.bin", std::ios::out | std::ofstream::binary);
		FILE.write(reinterpret_cast<const char*>(m_OutputBuffer), inferenceInfo.xSize * inferenceInfo.ySize * (inferenceInfo.zSize / 2) * sizeof(unsigned char));
		FILE.close();

		std::ofstream FILE2("./OutData/outputData_3DPVolume.bin", std::ios::out | std::ofstream::binary);
		FILE2.write(reinterpret_cast<const char*>(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.ySize * (inferenceInfo.zSize / 2)), inferenceInfo.xSize * inferenceInfo.ySize * (inferenceInfo.zSize / 2) * sizeof(unsigned char));
		FILE2.close();

		CString str;
		str.Format(_T("%d"), inferenceInfo.inferenceTime);
		MessageBox(L"Inference Done, Inference Time = " + str + "ms");
	}
	else
	{
		MessageBox(L"Inference Failed");
	}
}


void CMartianAIHostDlg::OnBnClickedButtonMode2()
{
	CString strFile;
	GetDlgItemText(IDC_EDIT_FILE2, strFile);
	CString strInVolumeSize;
	GetDlgItemText(IDC_EDIT_SEG2SIZE, strInVolumeSize);
	int inVolumeSize = _ttoi(strInVolumeSize);

	int size = inVolumeSize * inVolumeSize * inVolumeSize;
	int outSize = 128 * 128 * 128;

	std::ifstream fin(strFile, std::ios::binary);

	if (!fin)
	{
		MessageBox(L"Check File Name");
		return;
	}

	// 한번에 Loading
	fin.read(reinterpret_cast<char*>(m_InputBuffer), size * sizeof(unsigned char));

	InferenceInfo inferenceInfo;
	inferenceInfo.InferenceMode = MARTIAN_SEG_FETUS_2ND;
	inferenceInfo.xSize = inVolumeSize;
	inferenceInfo.ySize = inVolumeSize;
	inferenceInfo.zSize = inVolumeSize;
	memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
	memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
	inferenceInfo.labelNum = 8;

	//Gaussian Smooth Filter On - 1, Off - 0
	inferenceInfo.gaussianFilterMode = 0;

	// Dilation - 1, Erosion - 2, Both - 3
	inferenceInfo.morphologyMode = 0;
	// Dilation - 1, Erosion - 2, Both - 3
	inferenceInfo.processOnTable[1] = 2;	// 1 fluid   
	inferenceInfo.processOnTable[2] = 1;	// 2 face    
	inferenceInfo.processOnTable[3] = 1;	// 3 body    
	inferenceInfo.processOnTable[4] = 2;	// 4 ucord	
	inferenceInfo.processOnTable[5] = 1;	// 5 limbs
	inferenceInfo.processOnTable[6] = 0;	// 6 uterus	
	inferenceInfo.processOnTable[7] = 0;	// 7 placenta

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
		m_PlaneSize = inferenceInfo.xSize;
		m_PixelRatio = 1;

		SliderSetting(1, 0, inferenceInfo.xSize, inferenceInfo.xSize / 2);

		int plane = 64;
		CImage outImage1;
		outImage1.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);
		CImage outImage2;
		outImage2.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);
		CImage outImage3;
		outImage3.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);

		COLORREF px1;
		COLORREF px2;
		COLORREF px3;

		unsigned char RR1, GG1, BB1;
		unsigned char RR2, GG2, BB2;
		unsigned char RR3, GG3, BB3;

		for (int i = 0; i < inferenceInfo.xSize; i++)
		{
			for (int k = 0; k < inferenceInfo.xSize; k++)
			{
				RR2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);
				GG2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);
				BB2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);

				RR1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);
				GG1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);
				BB1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);

				RR3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);
				GG3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);
				BB3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);

				px1 = RGB(RR1, GG1, BB1);
				px2 = RGB(RR2, GG2, BB2);
				px3 = RGB(RR3, GG3, BB3);

				outImage1.SetPixel(i, k, px1);
				outImage2.SetPixel(i, k, px2);
				outImage3.SetPixel(i, k, px3);

			}
		}

		CRect rect1;
		m_picture1_control.GetWindowRect(rect1);
		CDC* dc1;
		dc1 = m_picture1_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc1->m_hDC, HALFTONE);
		outImage1.StretchBlt(dc1->m_hDC, 0, 0, rect1.Width(), rect1.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc1);//DC 해제

		CRect rect2;
		m_picture2_control.GetWindowRect(rect2);
		CDC* dc2;
		dc2 = m_picture2_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc2->m_hDC, HALFTONE);
		outImage2.StretchBlt(dc2->m_hDC, 0, 0, rect2.Width(), rect2.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc2);//DC 해제

		CRect rect3;
		m_picture3_control.GetWindowRect(rect3);
		CDC* dc3;
		dc3 = m_picture3_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc3->m_hDC, HALFTONE);
		outImage3.StretchBlt(dc3->m_hDC, 0, 0, rect3.Width(), rect3.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc3);//DC 해제

		std::ofstream FILE("./OutData/outputData_3DVolume.bin", std::ios::out | std::ofstream::binary);
		FILE.write(reinterpret_cast<const char*>(m_OutputBuffer), inferenceInfo.xSize * inferenceInfo.ySize * (inferenceInfo.zSize/2) * sizeof(unsigned char));
		FILE.close();

		std::ofstream FILE2("./OutData/outputData_3DPVolume.bin", std::ios::out | std::ofstream::binary);
		FILE2.write(reinterpret_cast<const char*>(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.ySize * (inferenceInfo.zSize / 2)), inferenceInfo.xSize * inferenceInfo.ySize * (inferenceInfo.zSize / 2) * sizeof(unsigned char));
		FILE2.close();

		CString str;
		str.Format(_T("%d"), inferenceInfo.inferenceTime);
		MessageBox(L"Inference Done, Inference Time = " + str + "ms");
	}
	else
	{
		MessageBox(L"Inference Failed");
	}
}


void CMartianAIHostDlg::OnBnClickedButtonMode3()
{
	CString strFile;
	GetDlgItemText(IDC_EDIT_FILE3, strFile);
	CString strInVolumeSize;
	GetDlgItemText(IDC_EDIT_SEG3SIZE, strInVolumeSize);
	int inVolumeSize = _ttoi(strInVolumeSize);

	int size = inVolumeSize * inVolumeSize * inVolumeSize;
	int outSize = 128 * 128 * 128;

	std::ifstream fin(strFile, std::ios::binary);

	if (!fin)
	{
		MessageBox(L"Check File Name");
		return;
	}
	// 한번에 Loading
	fin.read(reinterpret_cast<char*>(m_InputBuffer), size * sizeof(unsigned char));

	InferenceInfo inferenceInfo;
	inferenceInfo.InferenceMode = MARTIAN_CNS_SEG;
	inferenceInfo.xSize = inVolumeSize;
	inferenceInfo.ySize = inVolumeSize;
	inferenceInfo.zSize = inVolumeSize;
	memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
	memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
	inferenceInfo.labelNum = 8;

	//Gaussian Smooth Filter On - 1, Off - 0
	inferenceInfo.gaussianFilterMode = 0;

	// Dilation - 1, Erosion - 2, Both - 3
	inferenceInfo.morphologyMode = 0;
	// Dilation - 1, Erosion - 2, Both - 3
	inferenceInfo.processOnTable[1] = 0;	// 1 Th
	inferenceInfo.processOnTable[2] = 0;	// 2 CB
	inferenceInfo.processOnTable[3] = 0;	// 3 CM
	inferenceInfo.processOnTable[4] = 0;	// 4 CP
	inferenceInfo.processOnTable[5] = 0;	// 5 PVC
	inferenceInfo.processOnTable[6] = 0;	// 6 CSP
	inferenceInfo.processOnTable[7] = 0;	// 7 Midline

	inferenceInfo.thresholdTable[0] = 0.0; // 0 BG
	inferenceInfo.thresholdTable[1] = 0.5; // 1 Th
	inferenceInfo.thresholdTable[2] = 0.5; // 2 CB
	inferenceInfo.thresholdTable[3] = 0.5; // 3 CM
	inferenceInfo.thresholdTable[4] = 0.5; // 4 CP
	inferenceInfo.thresholdTable[5] = 0.5; // 5 PVC
	inferenceInfo.thresholdTable[6] = 0.5; // 6 CSP
	inferenceInfo.thresholdTable[7] = 0.5; // 7 MidLine

	int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

	if (resultValue == 0 && inferenceInfo.errorCode == 300)
	{
		m_PlaneSize = inferenceInfo.xSize;
		m_PixelRatio = 30;

		SliderSetting(1, 0, inferenceInfo.xSize, inferenceInfo.xSize / 2);

		int plane = 64;
		CImage outImage1;
		outImage1.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);
		CImage outImage2;
		outImage2.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);
		CImage outImage3;
		outImage3.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);

		COLORREF px1;
		COLORREF px2;
		COLORREF px3;

		unsigned char RR1, GG1, BB1;
		unsigned char RR2, GG2, BB2;
		unsigned char RR3, GG3, BB3;

		for (int i = 0; i < inferenceInfo.xSize; i++)
		{
			for (int k = 0; k < inferenceInfo.xSize; k++)
			{
				RR2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);
				GG2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);
				BB2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);

				RR1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);
				GG1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);
				BB1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);

				RR3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);
				GG3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);
				BB3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);

				px1 = RGB(RR1, GG1, BB1) * m_PixelRatio;
				px2 = RGB(RR2, GG2, BB2) * m_PixelRatio;
				px3 = RGB(RR3, GG3, BB3) * m_PixelRatio;

				outImage1.SetPixel(i, k, px1);
				outImage2.SetPixel(i, k, px2);
				outImage3.SetPixel(i, k, px3);
			}
		}

		CRect rect1;
		m_picture1_control.GetWindowRect(rect1);
		CDC* dc1;
		dc1 = m_picture1_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc1->m_hDC, HALFTONE);
		outImage1.StretchBlt(dc1->m_hDC, 0, 0, rect1.Width(), rect1.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc1);//DC 해제

		CRect rect2;
		m_picture2_control.GetWindowRect(rect2);
		CDC* dc2;
		dc2 = m_picture2_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc2->m_hDC, HALFTONE);
		outImage2.StretchBlt(dc2->m_hDC, 0, 0, rect2.Width(), rect2.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc2);//DC 해제

		CRect rect3;
		m_picture3_control.GetWindowRect(rect3);
		CDC* dc3;
		dc3 = m_picture3_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc3->m_hDC, HALFTONE);
		outImage3.StretchBlt(dc3->m_hDC, 0, 0, rect3.Width(), rect3.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc3);//DC 해제

		CString strFileOut1;
		strFileOut1 = "./OutData/";
		CString strFileOut2;
		strFileOut2 = strFileOut1 + strFile;

		std::ofstream FILE(strFileOut2, std::ios::out | std::ofstream::binary);
		FILE.write(reinterpret_cast<const char*>(m_OutputBuffer), inferenceInfo.xSize * inferenceInfo.ySize * inferenceInfo.zSize * sizeof(unsigned char));
		FILE.close();

		// Store CNS PVolume
		// [
		//std::ofstream FILE(strFileOut2, std::ios::out | std::ofstream::binary);
		//FILE.write(reinterpret_cast<const char*>(m_OutputBuffer), inferenceInfo.xSize * inferenceInfo.ySize * (inferenceInfo.zSize / 2) * sizeof(unsigned char));
		//FILE.close();

		//std::ofstream FILE2("./OutData/outputData_3DPVolume.bin", std::ios::out | std::ofstream::binary);
		//FILE2.write(reinterpret_cast<const char*>(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.ySize * (inferenceInfo.zSize / 2)), inferenceInfo.xSize * inferenceInfo.ySize * (inferenceInfo.zSize / 2) * sizeof(unsigned char));
		//FILE2.close();
		// ]

		CString str;
		str.Format(_T("%d"), inferenceInfo.inferenceTime);

		CString StrMean[7];
		for (int i = 0; i < 7; i++)
		{
			StrMean[i].Format(_T("[%2f, %2f, %2f]"), inferenceInfo.SegMean[i * 3 + 3], inferenceInfo.SegMean[i * 3 + 4], inferenceInfo.SegMean[i * 3 + 5]);
		}

		CString vecout1, vecout2, vecout3, meanout;
		vecout1.Format(_T("[%2f, %2f, %2f]"), inferenceInfo.pcaVector[0], inferenceInfo.pcaVector[1], inferenceInfo.pcaVector[2]);
		vecout2.Format(_T("[%2f, %2f, %2f]"), inferenceInfo.pcaVector[3], inferenceInfo.pcaVector[4], inferenceInfo.pcaVector[5]);
		vecout3.Format(_T("[%2f, %2f, %2f]"), inferenceInfo.pcaVector[6], inferenceInfo.pcaVector[7], inferenceInfo.pcaVector[8]);
		meanout.Format(_T("[%2f, %2f, %2f]"), inferenceInfo.pcaMean[0], inferenceInfo.pcaMean[1], inferenceInfo.pcaMean[2]);

		CString str_ErrorCode, str_ReturnValue;
		str_ErrorCode.Format(_T("%d"), inferenceInfo.errorCode);
		str_ReturnValue.Format(_T("%d"), inferenceInfo.returnValue);
		MessageBox(L"Inference Done, Inference Time: " + str + "ms" + "\n"
			+ "-------------------------\n"
			+ "Th Mean: " + StrMean[0] + "\n" + "CB Mean: " + StrMean[1] + "\n" + "CM Mean: " + StrMean[2] + "\n"
			+ "CP Mean: " + StrMean[3] + "\n" + "PVC Mean: " + StrMean[4] + "\n" + "CSP Mean: " + StrMean[5] + "\n"
			+ "Midline Mean: " + StrMean[6] + "\n" + L"eVec1: " + vecout1 + "\n" + "eVec2: " + vecout2 + "\n"
			+ "eVec3; " + vecout3 + "\n"
			+ "-------------------------\n"
			+ "errorCode: " + str_ErrorCode + "\n"
			+ "returnValue: " + str_ReturnValue);
	}
	else
	{
		CString str_ErrorCode, str_ReturnValue;
		str_ErrorCode.Format(_T("%d"), inferenceInfo.errorCode);
		str_ReturnValue.Format(_T("%d"), inferenceInfo.returnValue);
		MessageBox(L"Inference Failed\n-------------------------\nerrorCode: "
			+ str_ErrorCode + "\n"
			+ "returnValue: " + str_ReturnValue);
	}
}

void CMartianAIHostDlg::OnBnClickedButtonMode4()
{
	int size = 512 * 512 * 3;

	CString strFile;
	GetDlgItemText(IDC_EDIT_FILE4, strFile);

	CImage InImage;
	InImage.Load(strFile);
	
	int ImageWidth = InImage.GetWidth();
	int ImageHeight = InImage.GetHeight();
	int ImageBPP = InImage.GetBPP(); //픽셀당 비트수
	int ImageWidthB = ImageWidth * (ImageBPP / 8); //영상폭의 바이트수
	int memSize = ImageHeight * ImageWidthB;  //영상의 바이트수

	memset(m_InputImageBuffer, 0, sizeof(BYTE) * memSize);
	memset(m_InputBuffer, 0, sizeof(BYTE) * memSize);

	for (int y = 0; y < ImageHeight; y++) {

		BYTE* srcImg = NULL; ;

		srcImg = (BYTE*)InImage.GetPixelAddress(0, y);

		memcpy(&m_InputImageBuffer[y * ImageWidthB], srcImg, ImageWidthB);
	}

	//BGR to RGB
	int convertIndex = 0;
	for (int ii = 0; ii < ImageHeight; ++ii)
	{
		for (int jj = 0; jj < ImageWidth; ++jj)
		{
			*(m_InputBuffer + convertIndex + 0) = *(m_InputImageBuffer + convertIndex + 2);
			*(m_InputBuffer + convertIndex + 1) = *(m_InputImageBuffer + convertIndex + 1);
			*(m_InputBuffer + convertIndex + 2) = *(m_InputImageBuffer + convertIndex + 0);
			convertIndex += 3;
		}
	}

	std::ofstream FILEin("00inputData_2DImage.bin", std::ios::out | std::ofstream::binary);
	FILEin.write(reinterpret_cast<const char*>(m_InputBuffer), 400 * 400 * 3 * sizeof(unsigned char));
	FILEin.close();

	InferenceInfo inferenceInfo;
	inferenceInfo.InferenceMode = MARTIAN_STYLE_TRANSFER_ENHANCEMENT;
	inferenceInfo.xSize = ImageWidth;
	inferenceInfo.ySize = ImageHeight;
	inferenceInfo.zSize = 3;
	memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
	memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
	inferenceInfo.gaussianFilterMode = 0;
	inferenceInfo.morphologyMode = 0;
	inferenceInfo.inferenceTime = 0;
	inferenceInfo.imageRotate = 0;

	for (int i = 0; i < 16; i++)
	{
		inferenceInfo.thresholdTable[i] = 0.5;
	}

	int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

	//RGB to BGR
	convertIndex = 0;
	for (int ii = 0; ii < ImageHeight; ++ii)
	{
		for (int jj = 0; jj < ImageWidth; ++jj)
		{
			*(m_OutputImageBuffer + convertIndex + 0) = *(m_OutputBuffer + convertIndex + 2);
			*(m_OutputImageBuffer + convertIndex + 1) = *(m_OutputBuffer + convertIndex + 1);
			*(m_OutputImageBuffer + convertIndex + 2) = *(m_OutputBuffer + convertIndex + 0);

			convertIndex += 3;
		}
	}

	if (resultValue == 0)
	{

		//std::ofstream FILEout("00inputData_2DImage_infer.bin", std::ios::out | std::ofstream::binary);
		//FILEout.write(reinterpret_cast<const char*>(m_OutputImageBuffer), 400 * 400 * 3 * sizeof(unsigned char));
		//FILEout.close();

		m_Slider1.EnableWindow(0);
		m_Slider2.EnableWindow(0);
		m_Slider3.EnableWindow(0);

		m_PixelRatio = 1;

		CImage outImage;
		outImage.Create(ImageWidth, ImageHeight, 24);
		::SetBitmapBits(outImage, ImageWidth * ImageHeight * 3, m_OutputImageBuffer);

		outImage.Save(_T("./OutData/outImage.bmp"), Gdiplus::ImageFormatBMP);

		CRect rect;
		m_picture_control.GetWindowRect(rect);
		CDC* dc;
		dc = m_picture_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc->m_hDC, HALFTONE);
		outImage.StretchBlt(dc->m_hDC, 0, 0, rect.Width(), rect.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc);//DC 해제

		std::ofstream FILE("./OutData/outputData_2DImage.bin", std::ios::out | std::ofstream::binary);
		FILE.write(reinterpret_cast<const char*>(m_OutputBuffer), 400 * 400 * 3 * sizeof(unsigned char));
		FILE.close();

		CString str;
		str.Format(_T("%d"), inferenceInfo.inferenceTime);
		MessageBox(L"Inference Done, Inference Time = " + str + "ms");
	}
	else
	{
		MessageBox(L"Inference Failed");
	}
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
}

void CMartianAIHostDlg::OnBnClickedButtonMode5()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	CString strFile;
	GetDlgItemText(IDC_EDIT_FILE5, strFile);
	std::ifstream fin(strFile, std::ios::binary);

	CString strInVolumeSize;
	GetDlgItemText(IDC_EDIT_CNSM1SIZE, strInVolumeSize);
	int inImageSize = _ttoi(strInVolumeSize);
	int imageWidth = inImageSize;
	int imageHeight = inImageSize;

	if (!fin)
	{
		MessageBox(L"Check File Name");
		return;
	}
	// 한번에 Loading
	fin.read(reinterpret_cast<char*>(m_InputBuffer), imageWidth * imageHeight * sizeof(unsigned char));

	InferenceInfo inferenceInfo;
	inferenceInfo.InferenceMode = MARTIAN_CNS_MEASURETC;
	inferenceInfo.xSize = imageWidth;
	inferenceInfo.ySize = imageHeight;
	inferenceInfo.zSize = 1;
	memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
	memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
	inferenceInfo.gaussianFilterMode = 0;
	inferenceInfo.morphologyMode = 0;
	inferenceInfo.inferenceTime = 0;
	inferenceInfo.imageRotate = 0;

	for (int i = 0; i < 16; i++)
	{
		inferenceInfo.thresholdTable[i] = 0.5;
	}

	int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

	if (resultValue == 0)
	{
		m_PlaneSize = inferenceInfo.xSize;
		m_PixelRatio = 50;

		m_Slider1.EnableWindow(0);
		m_Slider2.EnableWindow(0);
		m_Slider3.EnableWindow(0);

		CImage outImage1;
		outImage1.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);

		COLORREF px1;

		unsigned char RR1, GG1, BB1;

		for (int i = 0; i < inferenceInfo.ySize; i++)
		{
			for (int k = 0; k < inferenceInfo.xSize; k++)
			{
				RR1 = *(m_OutputBuffer + inferenceInfo.xSize * i + k);
				GG1 = *(m_OutputBuffer + inferenceInfo.xSize * i + k);
				BB1 = *(m_OutputBuffer + inferenceInfo.xSize * i + k);

				px1 = RGB(RR1, GG1, BB1) * m_PixelRatio;

				outImage1.SetPixel(k, i, px1);
			}
		}

		CRect rect;
		m_picture_control.GetWindowRect(rect);
		CDC* dc;
		dc = m_picture_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc->m_hDC, HALFTONE);
		outImage1.StretchBlt(dc->m_hDC, 0, 0, rect.Width(), rect.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc);//DC 해제

		std::ofstream FILE("./OutData/outputData_2DImage.bin", std::ios::out | std::ofstream::binary);
		FILE.write(reinterpret_cast<const char*>(m_OutputBuffer), 512 * 512 * sizeof(unsigned char));
		FILE.close();

		CString str;
		str.Format(_T("%d"), inferenceInfo.inferenceTime);
		MessageBox(L"Inference Done, Inference Time = " + str + "ms");
	}
	else
	{
		MessageBox(L"Inference Failed");
	}
}

void CMartianAIHostDlg::OnBnClickedButtonMode6()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	CString strInVolumeSize;
	GetDlgItemText(IDC_EDIT_CNSM2SIZE, strInVolumeSize);
	int inImageSize = _ttoi(strInVolumeSize);
	int imageWidth = inImageSize;
	int imageHeight = inImageSize;

	CString strFile;
	GetDlgItemText(IDC_EDIT_FILE6, strFile);
	std::ifstream fin(strFile, std::ios::binary);

	if (!fin)
	{
		MessageBox(L"Check File Name");
		return;
	}
	// 한번에 Loading
	fin.read(reinterpret_cast<char*>(m_InputBuffer), imageWidth * imageHeight * sizeof(unsigned char));

	InferenceInfo inferenceInfo;
	inferenceInfo.InferenceMode = MARTIAN_CNS_MEASURETT;
	inferenceInfo.xSize = imageWidth;
	inferenceInfo.ySize = imageHeight;
	inferenceInfo.zSize = 1;
	memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
	memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
	inferenceInfo.gaussianFilterMode = 0;
	inferenceInfo.morphologyMode = 0;
	inferenceInfo.inferenceTime = 0;
	inferenceInfo.imageRotate = 0;

	for (int i = 0; i < 16; i++)
	{
		inferenceInfo.thresholdTable[i] = 0.5;
	}

	int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

	if (resultValue == 0)
	{
		m_PlaneSize = inferenceInfo.xSize;
		m_PixelRatio = 50;

		m_Slider1.EnableWindow(0);
		m_Slider2.EnableWindow(0);
		m_Slider3.EnableWindow(0);

		CImage outImage1;
		outImage1.Create(inferenceInfo.xSize, inferenceInfo.ySize, 32);

		COLORREF px1;

		unsigned char RR1, GG1, BB1;

		for (int i = 0; i < inferenceInfo.ySize; i++)
		{
			for (int k = 0; k < inferenceInfo.xSize; k++)
			{
				RR1 = *(m_OutputBuffer + inferenceInfo.xSize * i + k);
				GG1 = *(m_OutputBuffer + inferenceInfo.xSize * i + k);
				BB1 = *(m_OutputBuffer + inferenceInfo.xSize * i + k);

				px1 = RGB(RR1, GG1, BB1) * m_PixelRatio;

				outImage1.SetPixel(k, i, px1);
			}
		}

		CRect rect;
		m_picture_control.GetWindowRect(rect);
		CDC* dc;
		dc = m_picture_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc->m_hDC, HALFTONE);
		outImage1.StretchBlt(dc->m_hDC, 0, 0, rect.Width(), rect.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc);//DC 해제

		std::ofstream FILE("./OutData/outputData_2DImage.bin", std::ios::out | std::ofstream::binary);
		FILE.write(reinterpret_cast<const char*>(m_OutputBuffer), inferenceInfo.xSize * inferenceInfo.ySize * sizeof(unsigned char));
		FILE.close();

		CString str;
		str.Format(_T("%d"), inferenceInfo.inferenceTime);
		MessageBox(L"Inference Done, Inference Time = " + str + "ms");
	}
	else
	{
		MessageBox(L"Inference Failed");
	}
}

void CMartianAIHostDlg::OnBnClickedButtonMode7()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	CString strInVolumeSize;
	GetDlgItemText(IDC_EDIT_CNSM3SIZE, strInVolumeSize);
	int inImageSize = _ttoi(strInVolumeSize);
	int imageWidth = inImageSize;
	int imageHeight = inImageSize;

	CString strFile;
	GetDlgItemText(IDC_EDIT_FILE7, strFile);
	std::ifstream fin(strFile, std::ios::binary);

	if (!fin)
	{
		MessageBox(L"Check File Name");
		return;
	}
	// 한번에 Loading
	fin.read(reinterpret_cast<char*>(m_InputBuffer), imageWidth * imageHeight * sizeof(unsigned char));

	InferenceInfo inferenceInfo;
	inferenceInfo.InferenceMode = MARTIAN_CNS_MEASURETV;
	inferenceInfo.xSize = imageWidth;
	inferenceInfo.ySize = imageHeight;
	inferenceInfo.zSize = 1;
	memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
	memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
	inferenceInfo.gaussianFilterMode = 0;
	inferenceInfo.morphologyMode = 0;
	inferenceInfo.inferenceTime = 0;
	inferenceInfo.imageRotate = 0;

	for (int i = 0; i < 16; i++)
	{
		inferenceInfo.thresholdTable[i] = 0.5;
	}

	int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

	if (resultValue == 0)
	{
		m_PlaneSize = inferenceInfo.xSize;
		m_PixelRatio = 50;

		m_Slider1.EnableWindow(0);
		m_Slider2.EnableWindow(0);
		m_Slider3.EnableWindow(0);

		CImage outImage1;
		outImage1.Create(inferenceInfo.xSize, inferenceInfo.ySize, 32);

		COLORREF px1;

		unsigned char RR1, GG1, BB1;

		for (int i = 0; i < inferenceInfo.ySize; i++)
		{
			for (int k = 0; k < inferenceInfo.xSize; k++)
			{
				RR1 = *(m_OutputBuffer + inferenceInfo.xSize * i + k);
				GG1 = *(m_OutputBuffer + inferenceInfo.xSize * i + k);
				BB1 = *(m_OutputBuffer + inferenceInfo.xSize * i + k);

				px1 = RGB(RR1, GG1, BB1) * m_PixelRatio;

				outImage1.SetPixel(k, i, px1);
			}
		}

		CRect rect;
		m_picture_control.GetWindowRect(rect);
		CDC* dc;
		dc = m_picture_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc->m_hDC, HALFTONE);
		outImage1.StretchBlt(dc->m_hDC, 0, 0, rect.Width(), rect.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc);//DC 해제

		std::ofstream FILE("./OutData/outputData_2DImage.bin", std::ios::out | std::ofstream::binary);
		FILE.write(reinterpret_cast<const char*>(m_OutputBuffer), inferenceInfo.xSize * inferenceInfo.ySize * sizeof(unsigned char));
		FILE.close();

		CString str;
		str.Format(_T("%d"), inferenceInfo.inferenceTime);
		MessageBox(L"Inference Done, Inference Time = " + str + "ms");
	}
	else
	{
		MessageBox(L"Inference Failed");
	}
}

void CMartianAIHostDlg::OnDestroy()
{

	delete martianAIControl;


	//delete wrapper;

	// TODO: 여기에 메시지 처리기 코드를 추가합니다.
}

void CMartianAIHostDlg::OnBnClickedButtonGaussian()
{

	CString strFile;
	GetDlgItemText(IDC_EDIT_FILEGAU, strFile);
	CString strKernelSize;
	GetDlgItemText(IDC_EDIT_KERNEL, strKernelSize);
	int kernelSize = _ttoi(strKernelSize);
	CString strVolumeSize;
	GetDlgItemText(IDC_EDIT_GFVSIZE, strVolumeSize);
	int volumeASize = _ttoi(strVolumeSize);

	int size = volumeASize * volumeASize * volumeASize;
	std::ifstream fin(strFile, std::ios::binary);

	if (!fin)
	{
		MessageBox(L"Check File Name");
		return;
	}
	
	fin.read(reinterpret_cast<char*>(m_InputBuffer), size * sizeof(unsigned char));

	InferenceInfo inferenceInfo;
	inferenceInfo.InferenceMode = MARTIAN_3DPROC_GAUSSION;
	inferenceInfo.xSize = volumeASize;
	inferenceInfo.ySize = volumeASize;
	inferenceInfo.zSize = volumeASize;
	memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
	memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
	inferenceInfo.gaussianFilterMode = 0;
	inferenceInfo.morphologyMode = 0;
	inferenceInfo.inferenceTime = 0;
	inferenceInfo.kernelSize = kernelSize;

	int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

	if (resultValue == 0)
	{
		m_PlaneSize = inferenceInfo.xSize;
		m_PixelRatio = 1;

		SliderSetting(1, 0, inferenceInfo.xSize, inferenceInfo.xSize / 2);

		int plane = 64;
		CImage outImage1;
		outImage1.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);
		CImage outImage2;
		outImage2.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);
		CImage outImage3;
		outImage3.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);

		COLORREF px1;
		COLORREF px2;
		COLORREF px3;

		unsigned char RR1, GG1, BB1;
		unsigned char RR2, GG2, BB2;
		unsigned char RR3, GG3, BB3;

		for (int i = 0; i < inferenceInfo.xSize; i++)
		{
			for (int k = 0; k < inferenceInfo.xSize; k++)
			{
				RR2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);
				GG2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);
				BB2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);

				RR1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);
				GG1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);
				BB1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);

				RR3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);
				GG3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);
				BB3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);

				px1 = RGB(RR1, GG1, BB1);
				px2 = RGB(RR2, GG2, BB2);
				px3 = RGB(RR3, GG3, BB3);

				outImage1.SetPixel(i, k, px1);
				outImage2.SetPixel(i, k, px2);
				outImage3.SetPixel(i, k, px3);

			}
		}

		CRect rect1;
		m_picture1_control.GetWindowRect(rect1);
		CDC* dc1;
		dc1 = m_picture1_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc1->m_hDC, HALFTONE);
		outImage1.StretchBlt(dc1->m_hDC, 0, 0, rect1.Width(), rect1.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc1);//DC 해제

		CRect rect2;
		m_picture2_control.GetWindowRect(rect2);
		CDC* dc2;
		dc2 = m_picture2_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc2->m_hDC, HALFTONE);
		outImage2.StretchBlt(dc2->m_hDC, 0, 0, rect2.Width(), rect2.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc2);//DC 해제

		CRect rect3;
		m_picture3_control.GetWindowRect(rect3);
		CDC* dc3;
		dc3 = m_picture3_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc3->m_hDC, HALFTONE);
		outImage3.StretchBlt(dc3->m_hDC, 0, 0, rect3.Width(), rect3.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc3);//DC 해제

		std::ofstream FILE("./OutData/GaussianFilteredData.bin", std::ios::out | std::ofstream::binary);
		FILE.write(reinterpret_cast<const char*>(m_OutputBuffer), size * sizeof(unsigned char));
		FILE.close();

		CString str;
		str.Format(_T("%d"), inferenceInfo.inferenceTime);
		MessageBox(L"Inference Done, Inference Time = " + str + "ms");
	}
	else
	{
		MessageBox(L"Gaussian Filtering Failed");
	}



	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
}


void CMartianAIHostDlg::OnBnClickedButtonDilation()
{
	CString strFile;
	GetDlgItemText(IDC_EDIT_FILEDI, strFile);
	CString strLabelNum;
	CString strProcessLabel;
	GetDlgItemText(IDC_EDIT_LABELNUMDI, strLabelNum);
	GetDlgItemText(IDC_EDIT_PROCONDI, strProcessLabel);
	int LabelNum = _ttoi(strLabelNum);
	int ProcessLabel = _ttoi(strProcessLabel);

	CString strVolumeSize;
	GetDlgItemText(IDC_EDIT_DIVSIZE, strVolumeSize);
	int volumeASize = _ttoi(strVolumeSize);

	int size = volumeASize * volumeASize * volumeASize;
	std::ifstream fin(strFile, std::ios::binary);

	if (!fin)
	{
		MessageBox(L"Check File Name");
		return;
	}

	fin.read(reinterpret_cast<char*>(m_InputBuffer), size * sizeof(unsigned char));

	InferenceInfo inferenceInfo;
	inferenceInfo.InferenceMode = MARTIAN_3DPROC_DILATION;
	inferenceInfo.xSize = volumeASize;
	inferenceInfo.ySize = volumeASize;
	inferenceInfo.zSize = volumeASize;
	memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
	memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
	inferenceInfo.gaussianFilterMode = 0;
	inferenceInfo.morphologyMode = 0;
	inferenceInfo.inferenceTime = 0;
	inferenceInfo.labelNum = LabelNum;

	if (ProcessLabel != 0)
	{
		inferenceInfo.processOnTable[ProcessLabel - 1] = 1;
	}
	
	int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

	if (resultValue == 0)
	{
		m_PlaneSize = inferenceInfo.xSize;
		m_PixelRatio = 1;

		SliderSetting(1, 0, inferenceInfo.xSize, inferenceInfo.xSize / 2);

		int plane = 64;
		CImage outImage1;
		outImage1.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);
		CImage outImage2;
		outImage2.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);
		CImage outImage3;
		outImage3.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);

		COLORREF px1;
		COLORREF px2;
		COLORREF px3;

		unsigned char RR1, GG1, BB1;
		unsigned char RR2, GG2, BB2;
		unsigned char RR3, GG3, BB3;

		for (int i = 0; i < inferenceInfo.xSize; i++)
		{
			for (int k = 0; k < inferenceInfo.xSize; k++)
			{
				RR2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);
				GG2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);
				BB2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);

				RR1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);
				GG1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);
				BB1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);

				RR3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);
				GG3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);
				BB3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);

				px1 = RGB(RR1, GG1, BB1);
				px2 = RGB(RR2, GG2, BB2);
				px3 = RGB(RR3, GG3, BB3);

				outImage1.SetPixel(i, k, px1);
				outImage2.SetPixel(i, k, px2);
				outImage3.SetPixel(i, k, px3);

			}
		}

		CRect rect1;
		m_picture1_control.GetWindowRect(rect1);
		CDC* dc1;
		dc1 = m_picture1_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc1->m_hDC, HALFTONE);
		outImage1.StretchBlt(dc1->m_hDC, 0, 0, rect1.Width(), rect1.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc1);//DC 해제

		CRect rect2;
		m_picture2_control.GetWindowRect(rect2);
		CDC* dc2;
		dc2 = m_picture2_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc2->m_hDC, HALFTONE);
		outImage2.StretchBlt(dc2->m_hDC, 0, 0, rect2.Width(), rect2.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc2);//DC 해제

		CRect rect3;
		m_picture3_control.GetWindowRect(rect3);
		CDC* dc3;
		dc3 = m_picture3_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc3->m_hDC, HALFTONE);
		outImage3.StretchBlt(dc3->m_hDC, 0, 0, rect3.Width(), rect3.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc3);//DC 해제

		std::ofstream FILE("./OutData/DilationData.bin", std::ios::out | std::ofstream::binary);
		FILE.write(reinterpret_cast<const char*>(m_OutputBuffer), size * sizeof(unsigned char));
		FILE.close();

		CString str;
		str.Format(_T("%d"), inferenceInfo.inferenceTime);
		MessageBox(L"Inference Done, Inference Time = " + str + "ms");
	}
	else
	{
		MessageBox(L"Dilation Failed");
	}

	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
}


void CMartianAIHostDlg::OnBnClickedButtonErosion()
{
	CString strFile;
	GetDlgItemText(IDC_EDIT_FILEER, strFile);
	CString strLabelNum;
	CString strProcessLabel;

	GetDlgItemText(IDC_EDIT_LABELNUMER, strLabelNum);
	GetDlgItemText(IDC_EDIT_PROCONER, strProcessLabel);
	int LabelNum = _ttoi(strLabelNum);
	int ProcessLabel = _ttoi(strProcessLabel);
	CString strVolumeSize;
	GetDlgItemText(IDC_EDIT_ERVSIZE, strVolumeSize);
	int volumeASize = _ttoi(strVolumeSize);
	
	int size = volumeASize * volumeASize * volumeASize;
	std::ifstream fin(strFile, std::ios::binary);

	if (!fin)
	{
		MessageBox(L"Check File Name");
		return;
	}

	fin.read(reinterpret_cast<char*>(m_InputBuffer), size * sizeof(unsigned char));

	InferenceInfo inferenceInfo;
	inferenceInfo.InferenceMode = MARTIAN_3DPROC_EROSION;
	inferenceInfo.xSize = volumeASize;
	inferenceInfo.ySize = volumeASize;
	inferenceInfo.zSize = volumeASize;
	memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
	memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
	inferenceInfo.gaussianFilterMode = 0;
	inferenceInfo.morphologyMode = 0;
	inferenceInfo.inferenceTime = 0;
	inferenceInfo.labelNum = LabelNum;

	if (ProcessLabel != 0)
	{
		inferenceInfo.processOnTable[ProcessLabel - 1] = 2;
	}

	int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

	if (resultValue == 0)
	{
		m_PlaneSize = inferenceInfo.xSize;
		m_PixelRatio = 1;

		SliderSetting(1, 0, inferenceInfo.xSize, inferenceInfo.xSize / 2);

		int plane = 64;
		CImage outImage1;
		outImage1.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);
		CImage outImage2;
		outImage2.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);
		CImage outImage3;
		outImage3.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);

		COLORREF px1;
		COLORREF px2;
		COLORREF px3;

		unsigned char RR1, GG1, BB1;
		unsigned char RR2, GG2, BB2;
		unsigned char RR3, GG3, BB3;

		for (int i = 0; i < inferenceInfo.xSize; i++)
		{
			for (int k = 0; k < inferenceInfo.xSize; k++)
			{
				RR2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);
				GG2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);
				BB2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);

				RR1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);
				GG1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);
				BB1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);

				RR3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);
				GG3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);
				BB3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);

				px1 = RGB(RR1, GG1, BB1);
				px2 = RGB(RR2, GG2, BB2);
				px3 = RGB(RR3, GG3, BB3);

				outImage1.SetPixel(i, k, px1);
				outImage2.SetPixel(i, k, px2);
				outImage3.SetPixel(i, k, px3);

			}
		}

		CRect rect1;
		m_picture1_control.GetWindowRect(rect1);
		CDC* dc1;
		dc1 = m_picture1_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc1->m_hDC, HALFTONE);
		outImage1.StretchBlt(dc1->m_hDC, 0, 0, rect1.Width(), rect1.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc1);//DC 해제

		CRect rect2;
		m_picture2_control.GetWindowRect(rect2);
		CDC* dc2;
		dc2 = m_picture2_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc2->m_hDC, HALFTONE);
		outImage2.StretchBlt(dc2->m_hDC, 0, 0, rect2.Width(), rect2.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc2);//DC 해제

		CRect rect3;
		m_picture3_control.GetWindowRect(rect3);
		CDC* dc3;
		dc3 = m_picture3_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc3->m_hDC, HALFTONE);
		outImage3.StretchBlt(dc3->m_hDC, 0, 0, rect3.Width(), rect3.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc3);//DC 해제

		std::ofstream FILE("./OutData/ErosionData.bin", std::ios::out | std::ofstream::binary);
		FILE.write(reinterpret_cast<const char*>(m_OutputBuffer), size * sizeof(unsigned char));
		FILE.close();

		CString str;
		str.Format(_T("%d"), inferenceInfo.inferenceTime);
		MessageBox(L"Inference Done, Inference Time = " + str + "ms");
	}
	else
	{
		MessageBox(L"Erosion Failed");
	}



	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
}


void CMartianAIHostDlg::OnBnClickedButtonPca()
{
	CString strFile;
	CString strProcessLabel;
	CString strVolumeSize;
	CString strLabelNum;

	GetDlgItemText(IDC_EDIT_FILEPCA, strFile);
	GetDlgItemText(IDC_EDIT_PCALABEL, strProcessLabel);
	GetDlgItemText(IDC_EDIT_PCAVSIZE, strVolumeSize);
	GetDlgItemText(IDC_EDIT_LABELNUMPCA, strLabelNum);

	int PCATargetLabel = _ttoi(strProcessLabel);
	int volumeASize = _ttoi(strVolumeSize);
	int LabelNum = _ttoi(strLabelNum);

	int size = volumeASize * volumeASize * volumeASize;
	std::ifstream fin(strFile, std::ios::binary);

	if (!fin)
	{
		MessageBox(L"Check File Name");
		return;
	}

	fin.read(reinterpret_cast<char*>(m_InputBuffer), size * sizeof(unsigned char));

	InferenceInfo inferenceInfo;
	inferenceInfo.InferenceMode = MARTIAN_PROC_PCA;
	inferenceInfo.xSize = volumeASize;
	inferenceInfo.ySize = volumeASize;
	inferenceInfo.zSize = volumeASize;
	memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
	memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
	inferenceInfo.gaussianFilterMode = 0;
	inferenceInfo.morphologyMode = 0;
	inferenceInfo.inferenceTime = 0;
	inferenceInfo.labelNum = LabelNum;

	inferenceInfo.pcaTargetLabel = PCATargetLabel;

	int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

	if (resultValue == 0)
	{
		CString vecout1;
		vecout1.Format(_T("%2f, %2f, %2f"), inferenceInfo.pcaVector[0], inferenceInfo.pcaVector[1], inferenceInfo.pcaVector[2]);
		CString vecout2;
		vecout2.Format(_T("%2f, %2f, %2f"), inferenceInfo.pcaVector[3], inferenceInfo.pcaVector[4], inferenceInfo.pcaVector[5]);

		CString meanout;
		meanout.Format(_T("%2f, %2f, %2f"), inferenceInfo.pcaMean[0], inferenceInfo.pcaMean[1], inferenceInfo.pcaMean[2]);

		//CString str;
		//str.Format(_T("%d"), inferenceInfo.inferenceTime);
		MessageBox(L"eVec1 = " + vecout1 + "\n" + "eVec2 = " + vecout2 + "\n" + "mean = " + meanout);

	}
	else
	{
		MessageBox(L"PCA Failed");

	}



	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
}


void CMartianAIHostDlg::OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar)
{
	// TODO: 여기에 메시지 처리기 코드를 추가 및/또는 기본값을 호출합니다.

	CString temp_str;

	int nSPos = 0;
	if (pScrollBar)
	{
		if (pScrollBar == (CScrollBar*)&m_Slider1)
		{
			nSPos = m_Slider1.GetPos();
			temp_str.Format(_T("%d"), nSPos);
			m_Edit_Slider1.SetWindowTextW(temp_str);

			int plane = nSPos;
			CImage outImage;
			outImage.Create(m_PlaneSize, m_PlaneSize, 32);
			COLORREF px;

			unsigned char RR, GG, BB;
			for (int i = 0; i < m_PlaneSize; i++)
			{
				for (int k = 0; k < m_PlaneSize; k++)
				{
					//RR = *(m_OutputBuffer + m_PlaneSize * m_PlaneSize * plane + m_PlaneSize * k + i);
					//GG = *(m_OutputBuffer + m_PlaneSize * m_PlaneSize * plane + m_PlaneSize * k + i);
					//BB = *(m_OutputBuffer + m_PlaneSize * m_PlaneSize * plane + m_PlaneSize * k + i);

					RR = *(m_OutputBuffer + plane + m_PlaneSize * k + m_PlaneSize * m_PlaneSize * i);
					GG = *(m_OutputBuffer + plane + m_PlaneSize * k + m_PlaneSize * m_PlaneSize * i);
					BB = *(m_OutputBuffer + plane + m_PlaneSize * k + m_PlaneSize * m_PlaneSize * i);

					//RR = *(m_OutputBuffer + m_PlaneSize * plane + k + i * m_PlaneSize * m_PlaneSize);
					//GG = *(m_OutputBuffer + m_PlaneSize * plane + k + i * m_PlaneSize * m_PlaneSize);
					//BB = *(m_OutputBuffer + m_PlaneSize * plane + k + i * m_PlaneSize * m_PlaneSize);

					px = RGB(RR, GG, BB) * m_PixelRatio;
					outImage.SetPixel(i, k, px);
				}
			}

			CRect rect;
			m_picture1_control.GetWindowRect(rect);
			CDC* dc;
			dc = m_picture1_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
			//outImage.Draw(dc->m_hDC, 0, 0, outImage.GetWidth(), outImage.GetHeight());
			SetStretchBltMode(dc->m_hDC, HALFTONE);
			outImage.StretchBlt(dc->m_hDC, 0, 0, rect.Width(), rect.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
			ReleaseDC(dc);//DC 해제
		}
		if (pScrollBar == (CScrollBar*)&m_Slider2)
		{
			nSPos = m_Slider2.GetPos();
			temp_str.Format(_T("%d"), nSPos);
			m_Edit_Slider2.SetWindowTextW(temp_str);

			int plane = nSPos;
			CImage outImage;
			outImage.Create(m_PlaneSize, m_PlaneSize, 32);
			COLORREF px;

			unsigned char RR, GG, BB;
			for (int i = 0; i < m_PlaneSize; i++)
			{
				for (int k = 0; k < m_PlaneSize; k++)
				{
					RR = *(m_OutputBuffer + m_PlaneSize * m_PlaneSize * plane + m_PlaneSize * k + i);
					GG = *(m_OutputBuffer + m_PlaneSize * m_PlaneSize * plane + m_PlaneSize * k + i);
					BB = *(m_OutputBuffer + m_PlaneSize * m_PlaneSize * plane + m_PlaneSize * k + i);

					//RR = *(m_OutputBuffer + plane + m_PlaneSize * k + m_PlaneSize * m_PlaneSize * i);
					//GG = *(m_OutputBuffer + plane + m_PlaneSize * k + m_PlaneSize * m_PlaneSize * i);
					//BB = *(m_OutputBuffer + plane + m_PlaneSize * k + m_PlaneSize * m_PlaneSize * i);

					//RR = *(m_OutputBuffer + m_PlaneSize * plane + k + i * m_PlaneSize * m_PlaneSize);
					//GG = *(m_OutputBuffer + m_PlaneSize * plane + k + i * m_PlaneSize * m_PlaneSize);
					//BB = *(m_OutputBuffer + m_PlaneSize * plane + k + i * m_PlaneSize * m_PlaneSize);

					px = RGB(RR, GG, BB) * m_PixelRatio;
					outImage.SetPixel(i, k, px);
				}
			}

			CRect rect;
			m_picture2_control.GetWindowRect(rect);
			CDC* dc;
			dc = m_picture2_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
			//outImage.Draw(dc->m_hDC, 0, 0, outImage.GetWidth(), outImage.GetHeight());
			SetStretchBltMode(dc->m_hDC, HALFTONE);
			outImage.StretchBlt(dc->m_hDC, 0, 0, rect.Width(), rect.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
			ReleaseDC(dc);//DC 해제
		}
		if (pScrollBar == (CScrollBar*)&m_Slider3)
		{
			nSPos = m_Slider3.GetPos();
			temp_str.Format(_T("%d"), nSPos);
			m_Edit_Slider3.SetWindowTextW(temp_str);

			int plane = nSPos;
			CImage outImage;
			outImage.Create(m_PlaneSize, m_PlaneSize, 32);
			COLORREF px;

			unsigned char RR, GG, BB;
			for (int i = 0; i < m_PlaneSize; i++)
			{
				for (int k = 0; k < m_PlaneSize; k++)
				{
					//RR = *(m_OutputBuffer + m_PlaneSize * m_PlaneSize * plane + m_PlaneSize * k + i);
					//GG = *(m_OutputBuffer + m_PlaneSize * m_PlaneSize * plane + m_PlaneSize * k + i);
					//BB = *(m_OutputBuffer + m_PlaneSize * m_PlaneSize * plane + m_PlaneSize * k + i);

					//RR = *(m_OutputBuffer + plane + m_PlaneSize * k + m_PlaneSize * m_PlaneSize * i);
					//GG = *(m_OutputBuffer + plane + m_PlaneSize * k + m_PlaneSize * m_PlaneSize * i);
					//BB = *(m_OutputBuffer + plane + m_PlaneSize * k + m_PlaneSize * m_PlaneSize * i);

					RR = *(m_OutputBuffer + m_PlaneSize * plane + k + i * m_PlaneSize * m_PlaneSize);
					GG = *(m_OutputBuffer + m_PlaneSize * plane + k + i * m_PlaneSize * m_PlaneSize);
					BB = *(m_OutputBuffer + m_PlaneSize * plane + k + i * m_PlaneSize * m_PlaneSize);

					px = RGB(RR, GG, BB) * m_PixelRatio;
					outImage.SetPixel(i, k, px);
				}
			}

			CRect rect;
			m_picture3_control.GetWindowRect(rect);
			CDC* dc;
			dc = m_picture3_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
			//outImage.Draw(dc->m_hDC, 0, 0, outImage.GetWidth(), outImage.GetHeight());
			SetStretchBltMode(dc->m_hDC, HALFTONE);
			outImage.StretchBlt(dc->m_hDC, 0, 0, rect.Width(), rect.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
			ReleaseDC(dc);//DC 해제
		}
	}



	CDialogEx::OnHScroll(nSBCode, nPos, pScrollBar);
}


BOOL CMartianAIHostDlg::PreTranslateMessage(MSG* pMsg)
{
	// TODO: 여기에 특수화된 코드를 추가 및/또는 기본 클래스를 호출합니다.

	if (pMsg->message == WM_KEYDOWN)
	{
		if (pMsg->wParam == VK_RETURN || pMsg->wParam == VK_ESCAPE)
		{   // 위 VK_RETURN은 Enter, VK_ESCAPE는 ESC을 의미함. 필요시 하나만 사용.
			return true;
		}
	}

	return CDialogEx::PreTranslateMessage(pMsg);
}

int CMartianAIHostDlg::SliderSetting(int enable, int min, int max, int set)
{
	CString temp_str;
	temp_str.Format(_T("%d"), set);

	m_Slider1.SetRange(min, max);
	m_Slider1.SetRangeMin(min);
	m_Slider1.SetRangeMax(max);

	m_Slider2.SetRange(min, max);
	m_Slider2.SetRangeMin(min);
	m_Slider2.SetRangeMax(max);

	m_Slider3.SetRange(min, max);
	m_Slider3.SetRangeMin(min);
	m_Slider3.SetRangeMax(max);

	m_Slider1.EnableWindow(enable);
	m_Slider1.SetPos(set);
	m_Edit_Slider1.SetWindowTextW(temp_str);

	m_Slider2.EnableWindow(enable);
	m_Slider2.SetPos(set);
	m_Edit_Slider2.SetWindowTextW(temp_str);

	m_Slider3.EnableWindow(enable);
	m_Slider3.SetPos(set);
	m_Edit_Slider3.SetWindowTextW(temp_str);

	return 0;
}


void CMartianAIHostDlg::OnBnClickedButtonModent()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	CString strFile;
	GetDlgItemText(IDC_EDIT_FILENT, strFile);
	CString strInVolumeSize;
	GetDlgItemText(IDC_EDIT_NTSIZE, strInVolumeSize);
	int inVolumeSize = _ttoi(strInVolumeSize);

	int size = inVolumeSize * inVolumeSize * inVolumeSize;
	int outSize = 128 * 128 * 128;

	std::ifstream fin(strFile, std::ios::binary);

	if (!fin)
	{
		MessageBox(L"Check File Name");
		return;
	}

	// 한번에 Loading
	fin.read(reinterpret_cast<char*>(m_InputBuffer), size * sizeof(unsigned char));

	InferenceInfo inferenceInfo;
	inferenceInfo.InferenceMode = MARTIAN_SEG_NT;
	inferenceInfo.xSize = inVolumeSize;
	inferenceInfo.ySize = inVolumeSize;
	inferenceInfo.zSize = inVolumeSize;
	memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
	memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
	inferenceInfo.labelNum = 7;

	//Gaussian Smooth Filter On - 1, Off - 0
	inferenceInfo.gaussianFilterMode = 0;

	// Dilation - 1, Erosion - 2, Both - 3
	inferenceInfo.morphologyMode = 0;
	// Dilation - 1, Erosion - 2, Both - 3
	inferenceInfo.processOnTable[1] = 0;        // 
	inferenceInfo.processOnTable[2] = 0;        // 
	inferenceInfo.processOnTable[3] = 0;        // 
	inferenceInfo.processOnTable[4] = 0;        // 
	inferenceInfo.processOnTable[5] = 0;        // 
	inferenceInfo.processOnTable[6] = 0;        // 

	inferenceInfo.thresholdTable[0] = 0.0;
	inferenceInfo.thresholdTable[1] = 0.0;
	inferenceInfo.thresholdTable[2] = 0.0;
	inferenceInfo.thresholdTable[3] = 0.0;
	inferenceInfo.thresholdTable[4] = 0.0;
	inferenceInfo.thresholdTable[5] = 0.0;
	inferenceInfo.thresholdTable[6] = 0.0;

	int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

	if (resultValue == 0)
	{
		m_PlaneSize = inferenceInfo.xSize;
		m_PixelRatio = 30;

		SliderSetting(1, 0, inferenceInfo.xSize, inferenceInfo.xSize / 2);

		int plane = 64;
		CImage outImage1;
		outImage1.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);
		CImage outImage2;
		outImage2.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);
		CImage outImage3;
		outImage3.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);

		COLORREF px1;
		COLORREF px2;
		COLORREF px3;

		unsigned char RR1, GG1, BB1;
		unsigned char RR2, GG2, BB2;
		unsigned char RR3, GG3, BB3;

		for (int i = 0; i < inferenceInfo.xSize; i++)
		{
			for (int k = 0; k < inferenceInfo.xSize; k++)
			{
				RR2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);
				GG2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);
				BB2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);

				RR1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);
				GG1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);
				BB1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);

				RR3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);
				GG3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);
				BB3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);

				px1 = RGB(RR1, GG1, BB1) * m_PixelRatio;
				px2 = RGB(RR2, GG2, BB2) * m_PixelRatio;
				px3 = RGB(RR3, GG3, BB3) * m_PixelRatio;

				outImage1.SetPixel(i, k, px1);
				outImage2.SetPixel(i, k, px2);
				outImage3.SetPixel(i, k, px3);
			}
		}

		CRect rect1;
		m_picture1_control.GetWindowRect(rect1);
		CDC* dc1;
		dc1 = m_picture1_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc1->m_hDC, HALFTONE);
		outImage1.StretchBlt(dc1->m_hDC, 0, 0, rect1.Width(), rect1.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc1);//DC 해제

		CRect rect2;
		m_picture2_control.GetWindowRect(rect2);
		CDC* dc2;
		dc2 = m_picture2_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc2->m_hDC, HALFTONE);
		outImage2.StretchBlt(dc2->m_hDC, 0, 0, rect2.Width(), rect2.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc2);//DC 해제

		CRect rect3;
		m_picture3_control.GetWindowRect(rect3);
		CDC* dc3;
		dc3 = m_picture3_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc3->m_hDC, HALFTONE);
		outImage3.StretchBlt(dc3->m_hDC, 0, 0, rect3.Width(), rect3.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc3);//DC 해제

		std::ofstream FILE("./OutData/outputData_3DVolume.bin", std::ios::out | std::ofstream::binary);
		FILE.write(reinterpret_cast<const char*>(m_OutputBuffer), inferenceInfo.xSize * inferenceInfo.ySize * (inferenceInfo.zSize) * sizeof(unsigned char));
		FILE.close();

		CString str;
		str.Format(_T("%d"), inferenceInfo.inferenceTime);
		MessageBox(L"Inference Done, Inference Time = " + str + "ms");
	}
	else
	{
		MessageBox(L"Inference Failed");
	}
}


void CMartianAIHostDlg::OnBnClickedButtonEncryption()
{
	CString strFolder;
	GetDlgItemText(IDC_EDIT_FOLDERENCRYPTION, strFolder);
	string path = std::string(CT2CA(strFolder));
	string targetfiles = std::string(CT2CA(strFolder));
	targetfiles += "*.pt";

	struct _finddata_t fd;	
	intptr_t handle;	
	if ((handle = _findfirst(targetfiles.c_str(), &fd)) == -1L)
	{
		MessageBox(L"No pt file in directory!");
		return;
	}
	do 
	{
		char filename[256] = { 0, };

		memcpy(filename, path.c_str(), sizeof(char) * path.length());
		
		strcat_s(filename, fd.name);
		std::ifstream fin(filename, std::ios::binary);

		fin.seekg(0, ios::end);
		size_t length_load = fin.tellg();
		fin.seekg(0, ios::beg);
		unsigned char* modelBin = new unsigned char[length_load];
		fin.read(reinterpret_cast<char*>(modelBin), length_load * sizeof(unsigned char));

		unsigned char Key[] = { 0x76, 0x6f, 0x6c, 0x75, 0x6d, 0x65, 0x74, 0x72, 0x69, 0x63, 0x20, 0x61, 0x69, 0x6c, 0x61, 0x62 };

		unsigned char expandedKey[176] = { 0, };

		KeyExpansion(Key, expandedKey);

		int encryptionSize = ENCRYPTION_SIZE;
		if (length_load < encryptionSize)
		{
			encryptionSize = 16 * (int)(length_load / 16);
		}

		for (int i = 0; i < encryptionSize / 2; i += 16) {
			AESEncrypt(modelBin + i, expandedKey, modelBin + i);
		}
		for (int i = length_load - 16; i > (length_load - encryptionSize / 2); i -= 16) {
			AESEncrypt(modelBin + i, expandedKey, modelBin + i);
		}

		int filenameSize = strlen(filename);
		filename[filenameSize - 2] = 'm';
		filename[filenameSize - 1] = 'i';

		std::ofstream FILE(filename, std::ios::out | std::ofstream::binary);
		FILE.write(reinterpret_cast<const char*>(modelBin), length_load * sizeof(unsigned char));
		FILE.close();

		delete[] modelBin;
	} 
	while (_findnext(handle, &fd) == 0);	
	_findclose(handle);

	MessageBox(L"Encryption Done");

	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
}


void CMartianAIHostDlg::OnBnClickedButtonFaceclassification()
{
	CString strFile;
	GetDlgItemText(IDC_EDIT_FILE_FACECLASSIFICATION, strFile);

	CImage InImage;
	InImage.Load(strFile);

	int ImageWidth = InImage.GetWidth();
	int ImageHeight = InImage.GetHeight();
	int ImageBPP = InImage.GetBPP(); //픽셀당 비트수
	int ImageWidthB = ImageWidth * (ImageBPP / 8); //영상폭의 바이트수
	int memSize = ImageHeight * ImageWidthB;  //영상의 바이트수

	memset(m_InputImageBuffer, 0, sizeof(BYTE) * memSize);
	memset(m_InputBuffer, 0, sizeof(BYTE) * memSize);

	for (int y = 0; y < ImageHeight; y++) {

		BYTE* srcImg = NULL; ;

		srcImg = (BYTE*)InImage.GetPixelAddress(0, y);

		memcpy(&m_InputImageBuffer[y * ImageWidthB], srcImg, ImageWidthB);
	}

	//BGR to RGB
	int convertIndex = 0;
	for (int ii = 0; ii < ImageHeight; ++ii)
	{
		for (int jj = 0; jj < ImageWidth; ++jj)
		{
			*(m_InputBuffer + convertIndex + 0) = *(m_InputImageBuffer + convertIndex + 0);
			*(m_InputBuffer + convertIndex + 1) = *(m_InputImageBuffer + convertIndex + 1);
			*(m_InputBuffer + convertIndex + 2) = *(m_InputImageBuffer + convertIndex + 2);

			convertIndex += 3;
		}
	}

	std::ofstream FILEin("00inputData_2DImage.bin", std::ios::out | std::ofstream::binary);
	FILEin.write(reinterpret_cast<const char*>(m_InputBuffer), 400 * 400 * 3 * sizeof(unsigned char));
	FILEin.close();

	InferenceInfo inferenceInfo;
	inferenceInfo.InferenceMode = MARTIAN_STYLE_TRANSFER_FACECLASSIFICATION;
	inferenceInfo.xSize = ImageWidth;
	inferenceInfo.ySize = ImageHeight;
	inferenceInfo.zSize = 3;
	memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
	memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
	inferenceInfo.gaussianFilterMode = 0;
	inferenceInfo.morphologyMode = 0;
	inferenceInfo.inferenceTime = 0;
	inferenceInfo.imageRotate = 0;

	int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

	//RGB to BGR
	convertIndex = 0;
	for (int ii = 0; ii < ImageHeight; ++ii)
	{
		for (int jj = 0; jj < ImageWidth; ++jj)
		{
			*(m_OutputImageBuffer + convertIndex + 0) = *(m_InputBuffer + convertIndex + 0);
			*(m_OutputImageBuffer + convertIndex + 1) = *(m_InputBuffer + convertIndex + 1);
			*(m_OutputImageBuffer + convertIndex + 2) = *(m_InputBuffer + convertIndex + 2);

			convertIndex += 3;
		}
	}

	if (resultValue == 0)
	{
		m_Slider1.EnableWindow(0);
		m_Slider2.EnableWindow(0);
		m_Slider3.EnableWindow(0);

		m_PixelRatio = 1;

		CImage outImage;
		outImage.Create(ImageWidth, ImageHeight, 24);
		::SetBitmapBits(outImage, ImageWidth * ImageHeight * 3, m_OutputImageBuffer);

		CRect rect;
		m_picture_control.GetWindowRect(rect);
		CDC* dc;
		dc = m_picture_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc->m_hDC, HALFTONE);
		outImage.StretchBlt(dc->m_hDC, 0, 0, rect.Width(), rect.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc);//DC 해제

		CString str;
		str.Format(_T("%d"), inferenceInfo.inferenceTime);
		CString str_ErrorCode, str_ReturnValue;
		str_ReturnValue.Format(_T("%d"), inferenceInfo.errorCode);
		MessageBox(L"Inference Done, Inference Time: " + str + "ms" + "\n"
			+ "Classfication Result: " + str_ReturnValue);
	}
	else
	{
		MessageBox(L"Inference Failed");
	}

	
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
}

void CMartianAIHostDlg::OnBnClickedButtonModePelvicAssist()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	CString strFile;
	GetDlgItemText(IDC_EDIT_FILE_PELVICASSIST, strFile);
	CString strInVolumeSize;
	GetDlgItemText(IDC_EDIT_PELVICASSIST_SIZE, strInVolumeSize);
	int inVolumeSize = _ttoi(strInVolumeSize);

	int size = inVolumeSize * inVolumeSize * inVolumeSize;
	int outSize = 128 * 128 * 128;

	std::ifstream fin(strFile, std::ios::binary);

	if (!fin)
	{
		MessageBox(L"Check File Name");
		return;
	}

	// 한번에 Loading
	fin.read(reinterpret_cast<char*>(m_InputBuffer), size * sizeof(unsigned char));

	InferenceInfo inferenceInfo;
	inferenceInfo.InferenceMode = MARTIAN_SEG_PELVICASSIST;
	inferenceInfo.xSize = inVolumeSize;
	inferenceInfo.ySize = inVolumeSize;
	inferenceInfo.zSize = inVolumeSize;
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

	if (resultValue == 0)
	{
		m_PlaneSize = inferenceInfo.xSize;
		m_PixelRatio = 30;

		SliderSetting(1, 0, inferenceInfo.xSize, inferenceInfo.xSize / 2);

		int plane = 64;
		CImage outImage1;
		outImage1.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);
		CImage outImage2;
		outImage2.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);
		CImage outImage3;
		outImage3.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);

		COLORREF px1;
		COLORREF px2;
		COLORREF px3;

		unsigned char RR1, GG1, BB1;
		unsigned char RR2, GG2, BB2;
		unsigned char RR3, GG3, BB3;

		for (int i = 0; i < inferenceInfo.xSize; i++)
		{
			for (int k = 0; k < inferenceInfo.xSize; k++)
			{
				RR2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);
				GG2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);
				BB2 = *(m_OutputBuffer + inferenceInfo.xSize * inferenceInfo.xSize * plane + inferenceInfo.xSize * k + i);

				RR1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);
				GG1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);
				BB1 = *(m_OutputBuffer + plane + inferenceInfo.xSize * k + inferenceInfo.xSize * inferenceInfo.xSize * i);

				RR3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);
				GG3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);
				BB3 = *(m_OutputBuffer + inferenceInfo.xSize * plane + k + i * inferenceInfo.xSize * inferenceInfo.xSize);

				px1 = RGB(RR1, GG1, BB1) * m_PixelRatio;
				px2 = RGB(RR2, GG2, BB2) * m_PixelRatio;
				px3 = RGB(RR3, GG3, BB3) * m_PixelRatio;

				outImage1.SetPixel(i, k, px1);
				outImage2.SetPixel(i, k, px2);
				outImage3.SetPixel(i, k, px3);
			}
		}

		CRect rect1;
		m_picture1_control.GetWindowRect(rect1);
		CDC* dc1;
		dc1 = m_picture1_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc1->m_hDC, HALFTONE);
		outImage1.StretchBlt(dc1->m_hDC, 0, 0, rect1.Width(), rect1.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc1);//DC 해제

		CRect rect2;
		m_picture2_control.GetWindowRect(rect2);
		CDC* dc2;
		dc2 = m_picture2_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc2->m_hDC, HALFTONE);
		outImage2.StretchBlt(dc2->m_hDC, 0, 0, rect2.Width(), rect2.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc2);//DC 해제

		CRect rect3;
		m_picture3_control.GetWindowRect(rect3);
		CDC* dc3;
		dc3 = m_picture3_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc3->m_hDC, HALFTONE);
		outImage3.StretchBlt(dc3->m_hDC, 0, 0, rect3.Width(), rect3.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc3);//DC 해제

		std::ofstream FILE("./OutData/outputData_3DVolume.bin", std::ios::out | std::ofstream::binary);
		FILE.write(reinterpret_cast<const char*>(m_OutputBuffer), inferenceInfo.xSize * inferenceInfo.ySize * (inferenceInfo.zSize) * sizeof(unsigned char));
		FILE.close();

		CString str;
		str.Format(_T("%d"), inferenceInfo.inferenceTime);
		MessageBox(L"Inference Done, Inference Time = " + str + "ms");
	}
	else
	{
		MessageBox(L"Inference Failed");
	}
}

void CMartianAIHostDlg::OnBnClickedButtonModePelvicmeasure()
{
	CString strFile;
	GetDlgItemText(IDC_EDIT_FILE_PELVICASSISTMEASURE, strFile);
	CString strInVolumeSize;
	GetDlgItemText(IDC_EDIT_PELVICMEASURE_SIZE, strInVolumeSize);
	int inSize = _ttoi(strInVolumeSize);

	int size = inSize * inSize * 4;

	std::ifstream fin(strFile, std::ios::binary);

	if (!fin)
	{
		MessageBox(L"Check File Name");
		return;
	}

	// 한번에 Loading
	fin.read(reinterpret_cast<char*>(m_InputBuffer), size * sizeof(unsigned char));

	InferenceInfo inferenceInfo;
	inferenceInfo.InferenceMode = MARTIAN_SEG_PELVICMEASURE;
	inferenceInfo.xSize = inSize;
	inferenceInfo.ySize = inSize;
	inferenceInfo.zSize = 4;
	memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
	memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
	inferenceInfo.gaussianFilterMode = 0;
	inferenceInfo.morphologyMode = 0;
	inferenceInfo.inferenceTime = 0;
	inferenceInfo.imageRotate = 0;

	for (int i = 0; i < 16; i++)
	{
		inferenceInfo.thresholdTable[i] = 0.5;
	}

	int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

	if (resultValue == 0)
	{
		m_PlaneSize = inferenceInfo.xSize;
		m_PixelRatio = 1;

		m_Slider1.EnableWindow(0);
		m_Slider2.EnableWindow(0);
		m_Slider3.EnableWindow(0);

		CImage outImage1;
		outImage1.Create(inferenceInfo.xSize, inferenceInfo.xSize, 32);

		COLORREF px1;

		unsigned char RR1, GG1, BB1;

		for (int i = 0; i < inferenceInfo.ySize; i++)
		{
			for (int k = 0; k < inferenceInfo.xSize; k++)
			{
				RR1 = *(m_OutputBuffer + inferenceInfo.xSize * i + k);
				GG1 = *(m_OutputBuffer + inferenceInfo.xSize * i + k);
				BB1 = *(m_OutputBuffer + inferenceInfo.xSize * i + k);

				px1 = RGB(RR1, GG1, BB1) * m_PixelRatio;

				outImage1.SetPixel(k, i, px1);
			}
		}

		CRect rect;
		m_picture_control.GetWindowRect(rect);
		CDC* dc;
		dc = m_picture_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc->m_hDC, HALFTONE);
		outImage1.StretchBlt(dc->m_hDC, 0, 0, rect.Width(), rect.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc);//DC 해제

		std::ofstream FILE("./OutData/outputData_2DImage.raw", std::ios::out | std::ofstream::binary);
		FILE.write(reinterpret_cast<const char*>(m_OutputBuffer), inferenceInfo.xSize * inferenceInfo.ySize * sizeof(unsigned char));
		FILE.close();

		CString str;
		str.Format(_T("%d"), inferenceInfo.inferenceTime);
		MessageBox(L"Inference Done, Inference Time = " + str + "ms");
	}
	else
	{
		MessageBox(L"Inference Failed");
	}
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
}


void CMartianAIHostDlg::OnBnClickedButtonModentmeasure()
{

	CString strFile;
	GetDlgItemText(IDC_EDIT_FILENTMEASURE, strFile);

	CImage InImage;
	InImage.Load(strFile);

	int ImageWidth = InImage.GetWidth();
	int ImageHeight = InImage.GetHeight();
	int ImageBPP = InImage.GetBPP(); //픽셀당 비트수
	int ImageWidthB = ImageWidth * (ImageBPP / 8); //영상폭의 바이트수
	int memSize = ImageHeight * ImageWidthB;  //영상의 바이트수

	//memset(m_InputImageBuffer, 0, sizeof(BYTE) * memSize);
	memset(m_InputBuffer, 0, sizeof(BYTE) * memSize);

	for (int y = 0; y < ImageHeight; y++) {

		BYTE* srcImg = NULL; ;

		srcImg = (BYTE*)InImage.GetPixelAddress(0, y);

		memcpy(&m_InputBuffer[y * ImageWidthB], srcImg, ImageWidthB);
	}

	std::ofstream FILEin("00inputData_2DImage.raw", std::ios::out | std::ofstream::binary);
	FILEin.write(reinterpret_cast<const char*>(m_InputBuffer), ImageWidth * ImageHeight * 1 * sizeof(unsigned char));
	FILEin.close();

	InferenceInfo inferenceInfo;
	inferenceInfo.InferenceMode = MARTIAN_SEG_NTMEASURE;
	inferenceInfo.xSize = ImageWidth;
	inferenceInfo.ySize = ImageHeight;
	inferenceInfo.zSize = 1;
	memset(inferenceInfo.processOnTable, 0, sizeof(int) * 16);
	memset(inferenceInfo.thresholdTable, 0, sizeof(float) * 16);
	inferenceInfo.gaussianFilterMode = 0;
	inferenceInfo.morphologyMode = 0;
	inferenceInfo.inferenceTime = 0;
	inferenceInfo.imageRotate = 0;
	inferenceInfo.pcaVector;

	for (int i = 0; i < 16; i++)
	{
		inferenceInfo.thresholdTable[i] = 0.5;
	}

	int resultValue = martianAIControl->Run(m_InputBuffer, m_OutputBuffer, inferenceInfo);

	if (resultValue == 0)
	{
		m_PlaneSize = inferenceInfo.xSize;
		m_PixelRatio = 50;

		m_Slider1.EnableWindow(0);
		m_Slider2.EnableWindow(0);
		m_Slider3.EnableWindow(0);

		CImage outImage1;
		outImage1.Create(NT2DSEG_IMAGE_SIZE, NT2DSEG_IMAGE_SIZE, 32);

		COLORREF px1;

		unsigned char RR1, GG1, BB1;

		for (int i = 0; i < NT2DSEG_IMAGE_SIZE; i++)
		{
			for (int k = 0; k < NT2DSEG_IMAGE_SIZE; k++)
			{
				RR1 = *(m_OutputBuffer + NT2DSEG_IMAGE_SIZE * i + k);
				GG1 = *(m_OutputBuffer + NT2DSEG_IMAGE_SIZE * i + k);
				BB1 = *(m_OutputBuffer + NT2DSEG_IMAGE_SIZE * i + k);

				px1 = RGB(RR1, GG1, BB1) * m_PixelRatio;

				outImage1.SetPixel(k, i, px1);
			}
		}

		CRect rect;
		m_picture_control.GetWindowRect(rect);
		CDC* dc;
		dc = m_picture_control.GetDC(); //픽쳐 컨트롤의 DC를 얻는다.
		SetStretchBltMode(dc->m_hDC, HALFTONE);
		outImage1.StretchBlt(dc->m_hDC, 0, 0, rect.Width(), rect.Height(), SRCCOPY);//이미지를 픽쳐 컨트롤 크기로 조정
		ReleaseDC(dc);//DC 해제

		std::ofstream FILE("./OutData/outputData_2DImage.raw", std::ios::out | std::ofstream::binary);
		FILE.write(reinterpret_cast<const char*>(m_OutputBuffer), NT2DSEG_IMAGE_SIZE * NT2DSEG_IMAGE_SIZE * 2 * sizeof(unsigned char));
		FILE.close();

		CString str;
		str.Format(_T("%d"), inferenceInfo.inferenceTime);
		MessageBox(L"Inference Done, Inference Time = " + str + "ms");
	}
	else
	{
		MessageBox(L"Inference Failed");
	}
}
