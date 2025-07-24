#include "US3DLog.h"
#include <Shlwapi.h>

std::shared_ptr<spdlog::logger> _LogPreperator::_mLogger;

void _LogPreperator::Initialize(std::string project_name)
{
	if (_mLogger != nullptr) return;

	CHAR iniPath[MAX_PATH] = { 0, };
	if (GetModuleFileNameA(GetModuleHandleA(NULL), iniPath, MAX_PATH) < MAX_PATH)
	{
		PathRemoveFileSpecA(iniPath);
		PathAppendA(iniPath, "MartianLogInfo.ini");
	}

	CHAR level[20];
	DWORD numChar = GetPrivateProfileStringA("INFO", "level", "warning", level, 20, iniPath);
	UINT maxFiles = GetPrivateProfileIntA("INFO", "maxFiles", 30, iniPath);

	CHAR path[MAX_PATH];
	numChar = GetPrivateProfileStringA("INFO", "path", "./Martian.log", path, MAX_PATH, iniPath);
	if (numChar < 1) {
		return;
	}
	
	std::filesystem::path _path(path);
	std::string dir = _path.parent_path().string();
	char* parentPath = const_cast<char*>(dir.c_str());
	CreateDirectoryA(parentPath, NULL);

	std::ofstream ofs;
	ofs.open(path);
	if (ofs.fail())
	{
		strcpy_s(path, "./Martian.log");
	}
	ofs.close();

	_mLogger = spdlog::daily_logger_st(project_name.c_str(), path, 0, 0, false, static_cast<uint16_t>(maxFiles));

	spdlog::set_default_logger(_mLogger);
	spdlog::level::level_enum eLevel = spdlog::level::from_str(level);
	spdlog::set_pattern("[%D %T][%P][%n][%l][%s:%#:%!] %v");
	spdlog::info("Logger is ready! (level:{} ({}), maxFiles:{}, path:{}", level, eLevel, maxFiles, path);

	_mLogger->set_level(eLevel);
	spdlog::set_level(eLevel);
	spdlog::flush_every(std::chrono::seconds(1));
}

void _LogPreperator::Uninitialize()
{
	spdlog::drop_all();
}

_LogPreperator::~_LogPreperator()
{
}
