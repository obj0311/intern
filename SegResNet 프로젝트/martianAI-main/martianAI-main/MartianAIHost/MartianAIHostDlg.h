
// MartianAIHostDlg.h: 헤더 파일
//

#pragma once
#include <Windows.h>
#include <string>
#include <fstream>
#include <io.h>
#include "MartianAIControl.h"
#include "../MartianAI/Encryption.h"
#include "../MartianAI/MartianAIVar.h"

// CMartianAIHostDlg 대화 상자
class CMartianAIHostDlg : public CDialogEx
{
// 생성입니다.
public:
	CMartianAIHostDlg(CWnd* pParent = nullptr);	// 표준 생성자입니다.

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_MARTIANAIHOST_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 지원입니다.


// 구현입니다.
protected:
	HICON m_hIcon;

	// 생성된 메시지 맵 함수
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedButtonInit();
	afx_msg void OnBnClickedButtonMode1();
	afx_msg void OnBnClickedButtonMode2();
	afx_msg void OnBnClickedButtonMode3();
	afx_msg void OnBnClickedButtonMode4();
	unsigned char* m_InputBuffer = NULL;
	unsigned char* m_OutputBuffer = NULL;

	unsigned char* m_InputImageBuffer = NULL;
	unsigned char* m_OutputImageBuffer = NULL;
private:
	MartianAIControl* martianAIControl = NULL;

public:
	afx_msg void OnDestroy();
	afx_msg void OnBnClickedButtonGaussian();
	afx_msg void OnBnClickedButtonDilation();
	afx_msg void OnBnClickedButtonErosion();
	afx_msg void OnBnClickedButtonPca();

	CImage OutputImage;
	CStatic m_picture_control;
	CSliderCtrl m_Slider1;
	CEdit m_Edit_Slider1;
	afx_msg void OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar);

	int m_PlaneSize = 0;
	int m_PixelRatio = 1;
	CStatic m_picture1_control;
	CStatic m_picture2_control;
	CStatic m_picture3_control;
	CSliderCtrl m_Slider2;
	CSliderCtrl m_Slider3;
	CEdit m_Edit_Slider2;
	CEdit m_Edit_Slider3;
	virtual BOOL PreTranslateMessage(MSG* pMsg);
	int SliderSetting(int enable, int min, int max, int set);
	afx_msg void OnBnClickedButtonMode5();
	afx_msg void OnBnClickedButtonMode6();
	afx_msg void OnBnClickedButtonMode7();
	afx_msg void OnBnClickedButtonModent();
	afx_msg void OnBnClickedButtonModePelvicAssist();
	afx_msg void OnBnClickedButtonEncryption();
	afx_msg void OnBnClickedButtonFaceclassification();
	afx_msg void OnBnClickedButtonModePelvicmeasure();
	afx_msg void OnBnClickedButtonModentmeasure();
};
