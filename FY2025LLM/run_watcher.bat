@echo off
chcp 65001 > nul
echo [INFO] WSL 자동 동기화 감시 시작...
watchmedo shell-command ^
  --patterns="*" ^
  --recursive ^
  --verbose ^
  --command="bash \"/mnt/c/Users/litl/PycharmProjects/gitProject/StudyDev/FY2025LLM/sync_project.sh\"" ^
  C:/Users/litl/PycharmProjects/gitProject/StudyDev/FY2025LLM

pause
