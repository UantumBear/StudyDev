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



# utils/elastic/el_server_run.py
# import psutil, subprocess, time, socket, requests
# from requests.auth import HTTPBasicAuth

# def wait_for_port(host="localhost", port=9200, timeout=120):
#     start = time.time()
#     while time.time() - start < timeout:
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#             if s.connect_ex((host, port)) == 0:
#                 return True
#         time.sleep(1)
#     return False

# def wait_for_es_healthy(base_url, username=None, password=None, verify=False, timeout=120):
#     """/_cluster/health 가 yellow/green 될 때까지 대기"""
#     start = time.time()
#     auth = HTTPBasicAuth(username, password) if username and password else None
#     while time.time() - start < timeout:
#         try:
#             r = requests.get(f"{base_url}/_cluster/health", auth=auth, verify=verify, timeout=5)
#             if r.ok:
#                 status = r.json().get("status")
#                 if status in ("yellow", "green"):
#                     return True
#         except Exception:
#             pass
#         time.sleep(2)
#     return False

# def start_elasticsearch_server():
#     if is_port_in_use(9200) or is_elasticsearch_running():
#         print("[INFO] Elasticsearch가 이미 실행 중입니다.")
#         return

#     print("[INFO] Elasticsearch 서버 실행 중...")
#     subprocess.Popen(
#         [
#             "cmd.exe", "/c", "start",  # 새 콘솔
#             # 공백 경로인 경우를 대비해 따옴표 권장
#             r"C:\Users\litl\Elasticsearch\elasticsearch-8.14.1-windows-x86_64\elasticsearch-8.14.1\bin\elasticsearch.bat"
#         ],
#         shell=True,
#         creationflags=subprocess.CREATE_NEW_CONSOLE
#     )
#     print("[INFO] 실행 명령 전송, 포트 오픈 대기 중...")
#     if not wait_for_port("localhost", 9200, timeout=180):
#         raise RuntimeError("Elasticsearch 포트(9200) 오픈 대기 실패")

#     # HTTPS 기본 가정 (보안 활성화 기본값)
#     base_url = "https://localhost:9200"
#     username = os.getenv("ELASTIC_USER", "elastic")
#     password = os.getenv("ELASTIC_PASSWORD", "")
#     verify = os.getenv("ELASTIC_CA_CERT", "") or False  # CA 경로가 있으면 그 경로를 넣고, 없으면 False(경고는 뜸)

#     print("[INFO] 클러스터 헬스 대기 중...")
#     if not wait_for_es_healthy(base_url, username, password, verify=verify, timeout=180):
#         raise RuntimeError("Elasticsearch 클러스터 헬스 대기 실패")
#     print("[INFO] Elasticsearch 준비 완료")
