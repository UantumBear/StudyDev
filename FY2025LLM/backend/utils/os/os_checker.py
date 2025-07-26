import platform
import os
import sys

def check_os():
    system_name = platform.system()
    print(f"platform.system() 결과: {system_name}")

    if system_name == "Windows":
        print("현재 환경은 Windows입니다.")
    elif system_name == "Linux":
        print("현재 환경은 Linux입니다.")
    elif system_name == "Darwin":  # macOS
        print("현재 환경은 macOS입니다.")
    else:
        print("알 수 없는 운영체제입니다.")

    # 정보 출력 
    print(f"os.name: {os.name}")         # 'posix', 'nt', etc.
    print(f"sys.platform: {sys.platform}")  # 'win32', 'linux', 'darwin' 등

    return system_name

if __name__ == "__main__":
    check_os()

"""
python utilities/os/os_checker.py

"""