#pragma once

////
//  Copyright 2023 by Samsung Medison Co.,Ltd., All rights reserved.
//
//  This software is the confidential and proprietary information
//  of Samsung Medison, Inc. ("Confidential Information").
//  You shall not disclose such Confidential Information and shall use
//  it only in accordance with the terms of the license agreement
//  you entered into with Samsung Medison.
//
//  Notice!
//    1. Should be call "LOGGING_START" at the entry point of program.
//    2. Logger read the "LogInfo.ini" setting file, and initialize. (see .ini section)
//    3. Logger implement based on spdlog(https://github.com/gabime/spdlog) which under MIT license(tag v1.10.0).
//    4. spdlog available rich formatting using fmt library(https://github.com/fmtlib/fmt) which under MIT license.
//       so, if you want more rich formatting logging, see the fmtlib documents.
//    5. Do not call Logger macro & api directly except under 5 macros.
//       LOG_T();
//       LOG_D();
//       LOG_I();
//       LOG_W();
//       LOG_E();
//       LOG_C();
//
//  LogInfo.ini (it should be exist at same directory with exe file)
//    +------------------------------------------------------------------------+
//    |  [INFO]                                                                |
//    |  ; trace, debug, info, warning, error, critical, off                   |
//    |  level=trace                                                           |
//    |  ; maximum days for preserve the logging file.                         |
//    |  maxFiles=2                                                            |
//    |  ; log file path and naming rule                                       |
//    |  path=E:/logs/H25.log                                                  |
//    +------------------------------------------------------------------------+
//
//  ex)
//    LOG_T("This is trace log step {}", 1); // "This is trace log step 1"
//    LOG_D("Say hello world -> {}", "Hello World"); // "Say hello world -> Hello World"
//    LOG_I("Floating point precision under dot 2 : {:.2f}", 123.123123); // Floating point precision under dot 2 : 123.12
//    LOG_W("This is warn log {2}, {1}, {1}, {3}", "first", 2, "F"); // This is warn log 2, frist, first, F
//    LOG_E("This is error log");
//    LOG_C("This is critical log");
//
//

#pragma warning (disable: 4819)
#pragma warning (disable: 4244)
#pragma warning (disable: 4996)

#define SPDLOG_WCHAR_TO_UTF8_SUPPORT

#include <memory>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/daily_file_sink.h"
#include <fstream>
#include <filesystem>

#undef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#define LOG_T(...) _LogPreperator::Logger()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::trace, __VA_ARGS__)
#define LOG_D(...) _LogPreperator::Logger()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::debug, __VA_ARGS__)
#define LOG_I(...) _LogPreperator::Logger()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::info, __VA_ARGS__)
#define LOG_W(...) _LogPreperator::Logger()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::warn, __VA_ARGS__)
#define LOG_E(...) _LogPreperator::Logger()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::err, __VA_ARGS__)
#define LOG_C(...) _LogPreperator::Logger()->log(spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, spdlog::level::critical, __VA_ARGS__)

#define LOGGING_START(PROJNAME) _LogPreperator::Initialize(PROJNAME);
#define LOGGING_STOP _LogPreperator::Uninitialize();

class _LogPreperator {
public:
	static void Initialize(std::string project_name);
	static void Uninitialize();
	static std::shared_ptr<spdlog::logger> Logger() { return _mLogger; }

private:
	_LogPreperator();
	~_LogPreperator();

private:
	static std::shared_ptr<spdlog::logger> _mLogger;
};

#pragma warning (default: 4819)
#pragma warning (default: 4244)
#pragma warning (default: 4996)