@echo off
SET MartianAIProcessName=MartianAI.exe

tasklist /fi "ImageName eq %MartianAIProcessName%" /fo csv 2>NUL | find "%MartianAIProcessName%">NUL
if "%ERRORLEVEL%"=="0" (
  echo %MartianAIProcessName% is running , kill process
  taskkill /IM MartianAI.exe /F
) else (
  echo %MartianAIProcessName% is not running
)