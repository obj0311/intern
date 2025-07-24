SET MSBUILD_PATH="C:\Program Files\Microsoft Visual Studio\2022\Enterprise\MSBuild\Current\Bin"

@REM BUILD_OPTION
if /i "%3"=="build" (
SET MSBUILD_OPTION=/t:%3
) else (
SET MSBUILD_OPTION=/t:%3
)

@rem 2. Build
SET BUILD_PACKAGE=%WORKSPACE%\MartianAI.sln
set MSBUILD_CONFIGURATION=/p:configuration=Release;platform=x64
%MSBUILD_PATH%\MSBuild.exe %MSBUILD_OPTION% %MSBUILD_CONFIGURATION% %BUILD_PACKAGE%
IF %ERRORLEVEL%==0 (echo martianAI  Success) ELSE (EXIT)
