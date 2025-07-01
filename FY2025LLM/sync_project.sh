#!/bin/bash
# 프로젝트 동기화 스크립트

SRC_PATH="/mnt/c/Users/litl/PycharmProjects/gitProject/StudyDev/FY2025LLM"
DST_PATH="/home/devbear/dev_projects/StudyDev/FY2025LLM"
IGNORE_FILE="$SRC_PATH/.syncignore"

echo "[INFO] Windows → WSL 리눅스로 동기화 시작..."
rsync -av --exclude-from="$IGNORE_FILE" "$SRC_PATH/" "$DST_PATH/"
echo "[INFO] 복사 완료: $DST_PATH"


# 입력할 명령어
# 실행 권한 부여 (최초 1회만 필요)
# chmod +x /mnt/c/Users/litl/PycharmProjects/gitProject/StudyDev/FY2025LLM/sync_project.sh

# 실행 (매번 가능)
# bash /mnt/c/Users/litl/PycharmProjects/gitProject/StudyDev/FY2025LLM/sync_project.sh