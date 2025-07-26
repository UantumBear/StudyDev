@echo off
@chcp 65001 > nul

:: 날짜 + 시각 (예: 20250726_152430)
for /f %%i in ('powershell -command "Get-Date -Format yyyyMMdd_HHmmss"') do set NOW=%%i


:: 출력 디렉토리 설정
set OUTPUT_DIR=backend\conf\requirements

:: 디렉토리 생성 (없으면 자동 생성)
if not exist %OUTPUT_DIR% (
    mkdir %OUTPUT_DIR%
    echo [INFO] 디렉토리 생성: %OUTPUT_DIR%
)

:: 파일 경로 설정
set OUTPUT1=%OUTPUT_DIR%\requirements.txt
set OUTPUT2=%OUTPUT_DIR%\requirements_%NOW%.txt

:: 현재 경로 출력
echo [INFO] 현재 경로: %cd%
echo [INFO] 패키지 목록 추출 중...

:: pip freeze 실행
pip freeze > %OUTPUT1%
pip freeze > %OUTPUT2%

echo [DONE] 다음 파일이 생성되었습니다:
echo   - %OUTPUT1%
echo   - %OUTPUT2%

pause

:: backend/export_requirements.bat