#!/bin/bash
# 프로젝트 동기화 스크립트

SRC_PATH="/mnt/c/Users/litl/PycharmProjects/gitProject/StudyDev/FY2025LLM"
DST_PATH="/home/devbear/dev_projects/StudyDev/FY2025LLM"
IGNORE_FILE="$SRC_PATH/.syncignore"

echo "[INFO] Windows → WSL 리눅스로 동기화 시작..."
rsync -av --exclude-from="$IGNORE_FILE" "$SRC_PATH/" "$DST_PATH/"
echo "[INFO] 복사 완료: $DST_PATH"


# 수동 작업 시

# 입력할 명령어
# 실행 권한 부여 (최초 1회만 필요)
# chmod +x /mnt/c/Users/litl/PycharmProjects/gitProject/StudyDev/FY2025LLM/sync_project.sh

# 실행 (매번 가능)
# bash /mnt/c/Users/litl/PycharmProjects/gitProject/StudyDev/FY2025LLM/sync_project.sh

# 개발 중 자동 동기화

# 입력할 명령어
# [System Path] > pip install watchdog  ## 이미 있었다!
#
# 아래 명령어는 Windows PowerShell 에서 실행한다!
# watchmedo shell-command `
#   --patterns="*" `
#   --recursive `
#   --command='bash "/mnt/c/Users/litl/PycharmProjects/gitProject/StudyDev/FY2025LLM/sync_project.sh"' `
#   "C:/Users/litl/PycharmProjects/gitProject/StudyDev/FY2025LLM"


# watchmedo 가 C:/~ 경로를 감시해서 데이터가 변경된 경우 sync_project.sh 를 실행한다. 