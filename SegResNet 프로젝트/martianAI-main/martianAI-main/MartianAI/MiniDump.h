#pragma once 

/*
 *  Copyright 2013 by Samsung Co.,Ltd., All rights reserved.
 *
 *  This software is the confidential and proprietary information
 *  of Samsung, Inc. ("Confidential Information").  You
 *  shall not disclose such Confidential Information and shall use
 *  it only in accordance with the terms of the license agreement
 *  you entered into with Samsung.
 */

#include <WTypes.h>
#pragma warning(push)
#pragma warning(disable : 4091)
#include <DbgHelp.h>
#pragma warning(pop)

BOOL        BeginMiniDump(VOID);
VOID        EndMiniDump  (VOID);
LONG WINAPI UnHandledExceptionFilter(struct _EXCEPTOIN_POINTERS* exceptionInfo);
MINIDUMP_TYPE GetTypeMiniDump();