// SliceTo2DCore.cpp : Defines the entry point for the application.
//

#include "main.h"
#include "MiniDump.h"
#include "US3DLog.h"

#define MAX_LOADSTRING 100

// Global Variables:
HINSTANCE hInst;                                // current instance
WCHAR szTitle[MAX_LOADSTRING];                  // The title bar text
WCHAR szWindowClass[MAX_LOADSTRING];            // the main window class name

// Forward declarations of functions included in this code module:
ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);

#if !defined(ASAN_TEST)
void GetCudaVersion() {
	int cudaRuntimeVersion = 0;
	int cudaDriverVersion = 0;
	cudaError_t cudaError = cudaRuntimeGetVersion(&cudaRuntimeVersion);
	if (cudaError != cudaSuccess) {
		LOG_E("can not get Cuda Runtime Version");
	}
	else {
		LOG_I("Cuda Runtime Version:{}", cudaRuntimeVersion);
	}
	cudaError = cudaDriverGetVersion(&cudaDriverVersion);
	if (cudaError != cudaSuccess) {
		LOG_E("can not get Cuda Driver Version");
	}
	else {
		LOG_I("Cuda Driver Version:{}", cudaDriverVersion);
	}
}
#endif

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
	_In_opt_ HINSTANCE hPrevInstance,
	_In_ LPWSTR    lpCmdLine,
	_In_ int       nCmdShow)
{
	UNREFERENCED_PARAMETER(hPrevInstance);
	UNREFERENCED_PARAMETER(lpCmdLine);

#ifdef _DEBUG
	SetEnvironmentVariableA("OMP_NUM_THREADS", "1");
	SetEnvironmentVariableA("MKL_NUM_THREADS", "1");
	SetEnvironmentVariableA("KMP_DUPLICATE_LIB_OK", "TRUE");
#endif // _DEBUG

	SetEnvironmentVariable(L"CUDA_MODULE_LOADING", L"LAZY");

	LOGGING_START("MartianAI");
	BeginMiniDump();
#if !defined(ASAN_TEST)
	GetCudaVersion();
#endif	
	// Initialize global strings
	LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
	LoadStringW(hInstance, IDC_MARTIANAI, szWindowClass, MAX_LOADSTRING);
	MyRegisterClass(hInstance);

	// Perform application initialization:
	if (!InitInstance(hInstance, nCmdShow))
	{
		return FALSE;
	}

	HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_MARTIANAI));

	MSG msg;
	int breakcount = 0;
	// Main message loop:

	while (GetMessage(&msg, nullptr, 0, 0))
	{
		if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}
	EndMiniDump();
	LOGGING_STOP;
	return (int)msg.wParam;
}


//
//  FUNCTION: MyRegisterClass()
//
//  PURPOSE: Registers the window class.
//
ATOM MyRegisterClass(HINSTANCE hInstance)
{
	WNDCLASSEXW wcex;

	wcex.cbSize = sizeof(WNDCLASSEX);

	wcex.style = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc = WndProc;
	wcex.cbClsExtra = 0;
	wcex.cbWndExtra = 0;
	wcex.hInstance = hInstance;
	wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_MARTIANAI));
	wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
	wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex.lpszMenuName = 0;
	wcex.lpszClassName = szWindowClass;
	wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

	return RegisterClassExW(&wcex);
}

//
//   FUNCTION: InitInstance(HINSTANCE, int)
//
//   PURPOSE: Saves instance handle and creates main window
//
//   COMMENTS:
//
//        In this function, we save the instance handle in a global variable and
//        create and display the main program window.
//
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
	hInst = hInstance; // Store instance handle in our global variable

	HWND hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, nullptr, nullptr, hInstance, nullptr);

	if (!hWnd)
	{
		return FALSE;
	}

	ShowWindow(hWnd, nCmdShow);
	UpdateWindow(hWnd);

	return TRUE;
}

//
//  FUNCTION: WndProc(HWND, UINT, WPARAM, LPARAM)
//
//  PURPOSE:  Processes messages for the main window.
//
//  WM_COMMAND  - process the application menu
//  WM_PAINT    - Paint the main window
//  WM_DESTROY  - post a quit message and return
//
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
		case WM_COMMAND:
		{
			int wmId = LOWORD(wParam);
			//// Parse the menu selections:
			//switch (wmId)
			//{
			//case IDM_ABOUT:
			//	DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
			//	break;
			//case IDM_EXIT:
			//	DestroyWindow(hWnd);
			//	break;
			//default:
			//	return DefWindowProc(hWnd, message, wParam, lParam);
			//}
			break;
		}
		case WM_PAINT:
		{
			PAINTSTRUCT ps;
			HDC hdc = BeginPaint(hWnd, &ps);
			// TODO: Add any drawing code that uses hdc here...
			EndPaint(hWnd, &ps);
			break;
		}
		case WM_DESTROY:
		{
			PostQuitMessage(0);
			break;
		}
		case WM_COPYDATA:
		{
			if (wParam == (WPARAM)MESSAGE_INITIALIZE)
			{
				//LOG_T("MESSAGE_INITIALIZE Called");

				COPYDATASTRUCT* receiveData = (COPYDATASTRUCT*)lParam;


				InitSendInfo RecievedInfo;

				memcpy(&RecievedInfo, (COPYDATASTRUCT*)receiveData->lpData, receiveData->cbData);

				if (g_InferenceControl == NULL)
				{
					g_InferenceControl = new InferenceControl();
					LOG_T("InferenceControl Called");
					int returnvalue = g_InferenceControl->InitInference(RecievedInfo.m_InferenceDevics, RecievedInfo.m_DeviceNum, RecievedInfo.m_ModelPathLength, RecievedInfo.m_ModelPath);
					return returnvalue;
				}
				else
				{
					int returnvalue = g_InferenceControl->InitInference(RecievedInfo.m_InferenceDevics, RecievedInfo.m_DeviceNum, RecievedInfo.m_ModelPathLength, RecievedInfo.m_ModelPath);
					return returnvalue;
				}
			}
			else if (wParam == (WPARAM)MESSAGE_RUN)
			{
				LOG_T("MESSAGE_RUN Called");

				if (g_InferenceControl != NULL)
				{
					COPYDATASTRUCT* receiveData = (COPYDATASTRUCT*)lParam;
					int receivingParameter[2];

					memcpy_s(&receivingParameter, sizeof(int) * 2, receiveData->lpData, sizeof(int) * 2);

					LOG_T("receivingParameter[{}, {}]", receivingParameter[0], receivingParameter[1]);
					LOG_T("RunInference Call");

					int returnvalue = g_InferenceControl->RunInference(receivingParameter[0], receivingParameter[1]);

					LOG_T("RunInference returnvalue[{}]", returnvalue);

					return returnvalue;
				}
				else
				{
					return -1;
				}
			}
			break;
		}
		default:
		{
			return DefWindowProc(hWnd, message, wParam, lParam);
		}
	}
	return 0;
}

// Message handler for about box.
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
	UNREFERENCED_PARAMETER(lParam);
	switch (message)
	{
	case WM_INITDIALOG:
		return (INT_PTR)TRUE;

	case WM_COMMAND:
		if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
		{
			EndDialog(hDlg, LOWORD(wParam));
			return (INT_PTR)TRUE;
		}
		break;
	}
	return (INT_PTR)FALSE;
}
