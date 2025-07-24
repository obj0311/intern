/*
 *  Copyright 2013 by Samsung Co.,Ltd., All rights reserved.
 *
 *  This software is the confidential and proprietary information
 *  of Samsung, Inc. ("Confidential Information").  You
 *  shall not disclose such Confidential Information and shall use
 *  it only in accordance with the terms of the license agreement
 *  you entered into with Samsung.
 */
#define NOMINMAX

#include <Windows.h>
#include <tchar.h>
#include <sstream>
#include <Shlwapi.h>

#pragma warning(push)
#pragma warning(disable : 4091)
#include <DbgHelp.h>
#pragma warning(pop)

#include "MiniDump.h"

LPTOP_LEVEL_EXCEPTION_FILTER PreviousExceptionFilter = NULL;

typedef BOOL (WINAPI *MINIDUMPWRITEDUMP)	
(
	HANDLE		  hProcess,
	DWORD		  dwPid,
	HANDLE		  hFile,
	MINIDUMP_TYPE DumpType,

	CONST PMINIDUMP_EXCEPTION_INFORMATION	ExceptionParam,
	CONST PMINIDUMP_USER_STREAM_INFORMATION	UserStreamParam,
	CONST PMINIDUMP_CALLBACK_INFORMATION	CallbackParam
);

/**
 * @fn		BeginMiniDump
 * @brief	UnHandled Exception 발생시 덤프를 만들기 위해 
 *          사용자 정의한 UnHandledExceptionFilter()가 호출되도록 필터 등록.
 *          프로그램 시작 부분에서 호출
 */
BOOL BeginMiniDump(VOID)
{
	SetErrorMode(SEM_FAILCRITICALERRORS);

	PreviousExceptionFilter = SetUnhandledExceptionFilter((LPTOP_LEVEL_EXCEPTION_FILTER)UnHandledExceptionFilter);

	if (NULL == PreviousExceptionFilter)
		return FALSE;

	return TRUE;
}

/**
 * @fn		EndMiniDump
 * @brief	원래 필터 설정으로 복구하여 덤프가 종료되도록 설정 
 *          프로그램 종료 부분에서 호출
 */
VOID EndMiniDump  (VOID)
{
	SetUnhandledExceptionFilter(PreviousExceptionFilter);
}

/**
 * @fn		UnHandledExceptionFilter
 * @brief	Unhandled Exception 발생시 호출될 콜백 함수 
 *          DBGHELP.DLL 라이브러리 파일이 필요
 *			MiniDumpPath.ini에 설정한 경로에 Dump 파일을 생성
 *				[MiniDumpPath]
 *				Path =  D:/Logs/ProxyDump/
 *          Debug 모드일 경우 FullDump 생성
 *          Release 모드일 경우 MiniDump 생성
 */
LONG WINAPI UnHandledExceptionFilter(struct _EXCEPTOIN_POINTERS* exceptionInfo)
{
	HMODULE           hDebugHlpLib = NULL;
	MINIDUMPWRITEDUMP pfnDump      = NULL;
	HANDLE			  hDumpFile    = NULL;

	SYSTEMTIME SystemTime;
    TCHAR szDumpPath[MAX_PATH]    = { 0, };

	WCHAR iniPath[MAX_PATH] = { 0, };
	if (GetModuleFileNameW(GetModuleHandleA(NULL), iniPath, MAX_PATH) < MAX_PATH)
	{
		PathRemoveFileSpecW(iniPath);
		PathAppendW(iniPath, L"MartianDumpInfo.ini");
	}

	GetPrivateProfileString(L"MartianDumpPath", L"Path", L"", szDumpPath, 260, iniPath);

    TCHAR szDumpFName[MAX_PATH] = { 0, };
	
	_MINIDUMP_EXCEPTION_INFORMATION MiniDumpExceptionInfo;

	LONG nResult = EXCEPTION_CONTINUE_SEARCH;

	hDebugHlpLib = LoadLibrary(_T("DBGHELP.DLL"));

	if (hDebugHlpLib == NULL)
	{
		goto CleanUp;
	}
	
	pfnDump = (MINIDUMPWRITEDUMP)GetProcAddress(hDebugHlpLib, "MiniDumpWriteDump");

	if (pfnDump == NULL)
	{
		goto CleanUp;
	}	
	
	GetLocalTime(&SystemTime);	
		
	_sntprintf_s(szDumpFName, MAX_PATH, _TRUNCATE, _T("MartianAI_%d-%02d-%02d_%02d-%02d-%02d.dmp"),
		SystemTime.wYear,
		SystemTime.wMonth,
		SystemTime.wDay,
		SystemTime.wHour,
		SystemTime.wMinute,
		SystemTime.wSecond);

    BOOL ret = PathAppend(szDumpPath, szDumpFName);
    if (ret == FALSE)
    {
        memset(szDumpPath, 0, sizeof(TCHAR) * MAX_PATH);
        wcscpy_s(szDumpPath, szDumpFName);
    }

	hDumpFile = CreateFile(szDumpPath, GENERIC_WRITE, FILE_SHARE_WRITE, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

	if (hDumpFile == INVALID_HANDLE_VALUE)
	{
		goto CleanUp;
	}
	
	MiniDumpExceptionInfo.ThreadId			= GetCurrentThreadId();
	MiniDumpExceptionInfo.ExceptionPointers	= (PEXCEPTION_POINTERS)exceptionInfo;
	MiniDumpExceptionInfo.ClientPointers	= NULL;
		
	if ( pfnDump(
		GetCurrentProcess(),
		GetCurrentProcessId(),
		hDumpFile,			
		GetTypeMiniDump(),
		&MiniDumpExceptionInfo,	
		NULL,
		NULL))	
	{
		nResult = EXCEPTION_EXECUTE_HANDLER;
	}

CleanUp:
	if (hDumpFile)
		CloseHandle(hDumpFile);

	if (hDebugHlpLib)
		FreeLibrary(hDebugHlpLib);
		
	return nResult;
}

MINIDUMP_TYPE GetTypeMiniDump()
{
	char tempPath[_MAX_PATH] = {0, };
	GetCurrentDirectoryA(_MAX_PATH, tempPath);
	std::ostringstream oss;
	oss << tempPath << "\\pmud.ini";

	auto dumpmode = static_cast<MINIDUMP_TYPE>(GetPrivateProfileIntA("ejavm", "dhqtus", MiniDumpNormal, oss.str().c_str()));

	const auto developmentmode = GetPrivateProfileIntA("ejavm", "devel", 0, oss.str().c_str());
	if (developmentmode == 0)
	{
		if (dumpmode != MiniDumpNormal && dumpmode != MiniDumpWithFullMemory)
		{
			dumpmode = MiniDumpNormal;
		}
	}

	return dumpmode;
}