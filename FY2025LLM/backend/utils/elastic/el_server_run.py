# utils/elastic/el_server_run.py
import psutil
import subprocess
import time
import socket

def is_port_in_use(port: int) -> bool:
    """포트 사용 중인지 확인 (9200은 기본 Elasticsearch 포트)"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0

def is_elasticsearch_running() -> bool:
    """Elasticsearch 관련 Java 프로세스가 이미 실행 중인지 확인"""
    for proc in psutil.process_iter(attrs=["name", "cmdline"]):
        try:
            if "java" in proc.info["name"].lower():
                cmdline = " ".join(proc.info["cmdline"])
                if "elasticsearch" in cmdline.lower():
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

def start_elasticsearch_server():
    """Elasticsearch 서버 실행 (이미 실행 중이면 건너뜀)"""
    if is_port_in_use(9200):
        print("[INFO] 9200 포트가 이미 사용 중입니다. Elasticsearch가 실행 중일 수 있습니다.")
        return

    if is_elasticsearch_running():
        print("[INFO] Elasticsearch 프로세스가 이미 실행 중입니다.")
        return

    try:
        print("[INFO] Elasticsearch 서버 실행 중...")
        subprocess.Popen(
            [
             "cmd.exe",
             "/c",
             "start",
             "C:\\Users\\litl\\Elasticsearch\\elasticsearch-8.14.1-windows-x86_64\\elasticsearch-8.14.1\\bin\\elasticsearch.bat"
            ],
            shell=True,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        time.sleep(5)
        print("[INFO] Elasticsearch 실행 명령 완료")
    except Exception as e:
        print(f"[FAIL] Elasticsearch 실행 오류: {e}")
