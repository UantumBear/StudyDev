"""
model/hf_to_onnx.py
@역할 HuggingFace 타입 모델을 ONNX 타입 모델로 변환하는 함수

$ pip install transformers optimum[onnxruntime] onnx --upgrade
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

"""
import sys
import os
import subprocess
import onnxruntime as ort
print(ort.get_available_providers())

# ======================= 설정 ==========================
# 1. 변환 대상 모델 경로 또는 Hugging Face ID
model_path = "models/meta-llama/Llama-3.2-1B-Instruct"  # 또는 로컬 경로
local_files_only = True  # 인터넷 없이 로컬 모델만 사용할 경우 True

# 2. 변환 후 저장할 위치
output_dir = "models/meta-llama/Llama-3.2-1B-Instruct/onnx"
os.makedirs(output_dir, exist_ok=True)


# CLI 명령어 구성
command = [
    "optimum-cli", "export", "onnx",
    "--model", model_path,
    "--task", "text-generation",
    "--device", "cuda",  # GPU 사용, '--device'와 'cuda'를 별도의 인자로 분리해야 함
    "--dtype", "fp16",
    "--atol", "1e-3",   # 선택사항: 허용 오차
     output_dir           # 저장 경로 명시
]

# 명령어 실행
print(f"optimum-cli 명령어 실행 중: {' '.join(command)}")

# 실시간 로그를 출력 (버퍼링 없이)
process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
process.wait()

# result = subprocess.run(command, capture_output=True, text=True) # 모든 로그를 버퍼에 저장한 뒤 종료 시 한 번에 출력
# print("stdout:\n", result.stdout)
# print("stderr:\n", result.stderr)

if process.returncode == 0:
    print("성공적으로 ONNX로 변환되었습니다!")
else:
    print("변환 중 오류가 발생했습니다.")
    print("오류 코드:", process.returncode)
    print("stderr 메시지를 확인하여 추가 디버깅을 수행하세요.")

    print(f"\n{output_dir} 디렉토리 내용:")
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            print(" -", f)
    else:
        print("디렉토리가 존재하지 않습니다.")


# optimum-cli export onnx --help  ## 명령어 참고
# python model/hf_to_onnx.py
# ls -lh models/meta-llama/Llama3.2-1B-Instruction/onnx/model.onnx
