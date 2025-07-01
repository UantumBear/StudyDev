```shell
# (venv312) PS C:\Users\litl\PycharmProjects\gitProject\StudyDev\FY2025LLM\utils\pyhwp> 
git clone https://github.com/mete0r/pyhwp.git

(venv312) PS C:\Users\litl\PycharmProjects\gitProject\StudyDev\FY2025LLM\utils\pyhwp> cd pyhwp
# (venv312) PS C:\Users\litl\PycharmProjects\gitProject\StudyDev\FY2025LLM\utils\pyhwp\pyhwp> 
pip install .

# 결과:
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Processing c:\users\litl\pycharmprojects\gitproject\studydev\fy2025llm\utils\pyhwp\pyhwp
  Preparing metadata (setup.py) ... done
Requirement already satisfied: cryptography>=40.0.1 in c:\users\litl\pycharmprojects\gitproject\studydev\fy2025llm\venv312\lib\site-packages (from pyhwp==0.1b16.dev0) (45.0.4)
Requirement already satisfied: lxml>=4.9.2 in c:\users\litl\pycharmprojects\gitproject\studydev\fy2025llm\venv312\lib\site-packages (from pyhwp==0.1b16.dev0) (5.4.0)
Requirement already satisfied: olefile>=0.43 in c:\users\litl\pycharmprojects\gitproject\studydev\fy2025llm\venv312\lib\site-packages (from pyhwp==0.1b16.dev0) (0.47)
Requirement already satisfied: cffi>=1.14 in c:\users\litl\pycharmprojects\gitproject\studydev\fy2025llm\venv312\lib\site-packages (from cryptography>=40.0.1->pyhwp==0.1b16.dev0) (1.17.1)
Requirement already satisfied: pycparser in c:\users\litl\pycharmprojects\gitproject\studydev\fy2025llm\venv312\lib\site-packages (from cffi>=1.14->cryptography>=40.0.1->pyhwp==0.1b16.dev0) (2.22)
Building wheels for collected packages: pyhwp
  DEPRECATION: Building 'pyhwp' using the legacy setup.py bdist_wheel mechanism, which will be removed in a future version. pip 25.3 will enforce this behaviour change. A possible replacement is to
 use the standardized build interface by setting the `--use-pep517` option, (possibly combined with `--no-build-isolation`), or adding a `pyproject.toml` file to the source tree of 'pyhwp'. Discussion can be found at https://github.com/pypa/pip/issues/6334
  Building wheel for pyhwp (setup.py) ... done
  Created wheel for pyhwp: filename=pyhwp-0.1b16.dev0-py3-none-any.whl size=293361 sha256=ca76e16cbf9f579ac5346adf6b4d4d938ab701c6004ee3c63f852d11da9ef920
  Stored in directory: C:\Users\litl\AppData\Local\Temp\pip-ephem-wheel-cache-8w71pxql\wheels\c8\44\1a\08070d74aa2179cb5be8d79b8fd208304ec4cfc9501ca10acb
Successfully built pyhwp
Installing collected packages: pyhwp
Successfully installed pyhwp-0.1b16.dev0

```
# 참고
이 라이브러리는 pip install pyhwp 로 설치하고, CLI 를 이용해서 hwp 파일을 변환할 수 있도록 지원하는  
라이브러리이다.
즉 바로 python 의 function()  과 같이 제공하지 않기 때문에,  
CLI 명령을 실행시키는 함수를 만들어서 작동시켜야 한다.

25.06.22 일단 구버전 hwp parsing 은 실패했다. 5버전의 경우 추출 가능했다.


# 참고 link
https://wikidocs.net/170811
