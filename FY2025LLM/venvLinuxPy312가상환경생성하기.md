##### 1. Pycharm 에서 Ubuntu 터미널 접속


###### 외부 저장소(PPA) 목록 확인하기
이미 외부 저장소가 있는 지 확인한다.
```shell
ls /etc/apt/sources.list.d/
```

###### 외부 저장소(PPA) 등록하기 
만약 없다면, (Ubuntu에 기본적으로 제공하지 않는) 다양한 Python 버전(예: 3.7, 3.11, 3.12 등)을 설치하기 위해    
신뢰할 수 있는 외부 저장소를 등록해야 한다.  
```shell
sudo add-apt-repository ppa:deadsnakes/ppa
```
결과:Ubuntu PW 를 입력하라는 메시지가 떴다. 패스워드를 입력하자 아래와 같은 결과를 받았다.
```shell
Repository: 'deb https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu/ jammy main'
Description:
This PPA contains more recent Python versions packaged for Ubuntu.

Disclaimer: there's no guarantee of timely updates in case of security problems or other issues. If you want to use them in a security-or-otherwise-critical environment (say, on a production server), you do so at your own risk.

Update Note
===========
Please use this repository instead of ppa:fkrull/deadsnakes.

Reporting Issues
================

Issues can be reported in the master issue tracker at:
https://github.com/deadsnakes/issues/issues

Supported Ubuntu and Python Versions
====================================

- Ubuntu 20.04 (focal) Python3.5 - Python3.7, Python3.9 - Python3.13
- Ubuntu 22.04 (jammy) Python3.7 - Python3.9, Python3.11 - Python3.13
- Ubuntu 24.04 (noble) Python3.7 - Python3.11, Python3.13
- Note: Python2.7 (focal, jammy), Python 3.8 (focal), Python 3.10 (jammy), Python3.12 (noble) are not provided by deadsnakes as upstream ubuntu provides those packages.

Why some packages aren't built:
- Note: for focal, older python versions require libssl<1.1 so they are not currently built
- Note: for jammy and noble, older python versions requre libssl<3 so they are not currently built
- If you need these, reach out to asottile to set up a private ppa

The packages may also work on other versions of Ubuntu or Debian, but that is not tested or supported.

Packages
========

The packages provided here are loosely based on the debian upstream packages with some modifications to make them more usable as non-default pythons and on ubuntu.  As such, the packages follow debian's patterns and often do not include a full python distribution with just `apt install python#.#`.  Here is a list of packages that may be useful along with the default install:       

- `python#.#-dev`: includes development headers for building C extensions
- `python#.#-venv`: provides the standard library `venv` module
- `python#.#-distutils`: provides the standard library `distutils` module
- `python#.#-lib2to3`: provides the `2to3-#.#` utility as well as the standard library `lib2to3` module
- `python#.#-gdbm`: provides the standard library `dbm.gnu` module
- `python#.#-tk`: provides the standard library `tkinter` module

Third-Party Python Modules
==========================

Python modules in the official Ubuntu repositories are packaged to work with the Python interpreters from the official repositories. Accordingly, they generally won't work with the Python interpreters from this PPA. As an exception, pure-Python modules for Python 3 will work, but any compiled extension modules won't.

To install 3rd-party Python modules, you should use the common Python packaging tools.  For an introduction into the Python packaging ecosystem and its tools, refer to the Python Packaging User Guide:
https://packaging.python.org/installing/

Sources
=======
The package sources are available at:
https://github.com/deadsnakes/

Nightly Builds
==============

For nightly builds, see ppa:deadsnakes/nightly https://launchpad.net/~deadsnakes/+archive/ubuntu/nightly
More info: https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa
Adding repository.
Press [ENTER] to continue or Ctrl-c to cancel.
```
>> 결과:
Enter 를 입력하자 jammy 라는 파일들이 다운로드 되었다.
```shell
Adding deb entry to /etc/apt/sources.list.d/deadsnakes-ubuntu-ppa-jammy.list
Adding disabled deb-src entry to /etc/apt/sources.list.d/deadsnakes-ubuntu-ppa-jammy.list
Adding key to /etc/apt/trusted.gpg.d/deadsnakes-ubuntu-ppa.gpg with fingerprint F23C5A6CF475977595C89F51BA6932366A755776
Hit:1 http://archive.ubuntu.com/ubuntu jammy InRelease
Get:2 http://archive.ubuntu.com/ubuntu jammy-updates InRelease [128 kB]                                        
Get:3 http://security.ubuntu.com/ubuntu jammy-security InRelease [129 kB]                                       
Get:4 http://archive.ubuntu.com/ubuntu jammy-backports InRelease [127 kB]                                                  
Get:5 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy InRelease [18.1 kB]
Get:6 http://security.ubuntu.com/ubuntu jammy-security/main amd64 Packages [2410 kB]       
Get:7 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy/main amd64 Packages [28.0 kB]
Get:8 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 Packages [2659 kB]                   
Get:9 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy/main Translation-en [5176 B]
Get:10 http://security.ubuntu.com/ubuntu jammy-security/main Translation-en [363 kB]                     
Get:11 http://security.ubuntu.com/ubuntu jammy-security/main amd64 c-n-f Metadata [13.6 kB]
Get:12 http://security.ubuntu.com/ubuntu jammy-security/restricted amd64 Packages [3564 kB]
Get:13 http://archive.ubuntu.com/ubuntu jammy-updates/main Translation-en [428 kB]                 
Get:14 http://archive.ubuntu.com/ubuntu jammy-updates/main amd64 c-n-f Metadata [18.5 kB]         
Get:15 http://archive.ubuntu.com/ubuntu jammy-updates/restricted amd64 Packages [3695 kB]
Get:16 http://security.ubuntu.com/ubuntu jammy-security/restricted Translation-en [638 kB]           
Get:17 http://security.ubuntu.com/ubuntu jammy-security/restricted amd64 c-n-f Metadata [624 B]          
Get:18 http://security.ubuntu.com/ubuntu jammy-security/universe amd64 Packages [980 kB]
Get:19 http://security.ubuntu.com/ubuntu jammy-security/universe Translation-en [212 kB]           
Get:20 http://security.ubuntu.com/ubuntu jammy-security/universe amd64 c-n-f Metadata [21.7 kB]           
Get:21 http://security.ubuntu.com/ubuntu jammy-security/multiverse amd64 Packages [39.6 kB]
Get:22 http://archive.ubuntu.com/ubuntu jammy-updates/restricted Translation-en [658 kB]            
Get:23 http://security.ubuntu.com/ubuntu jammy-security/multiverse Translation-en [8716 B]              
Get:24 http://security.ubuntu.com/ubuntu jammy-security/multiverse amd64 c-n-f Metadata [368 B]
Get:25 http://archive.ubuntu.com/ubuntu jammy-updates/restricted amd64 c-n-f Metadata [676 B]                                                                                                  
Get:26 http://archive.ubuntu.com/ubuntu jammy-updates/universe amd64 Packages [1215 kB]                                                                                                        
Get:27 http://archive.ubuntu.com/ubuntu jammy-updates/universe Translation-en [300 kB]                                                                                                          
Get:28 http://archive.ubuntu.com/ubuntu jammy-updates/universe amd64 c-n-f Metadata [28.7 kB]                                                                                                   
Get:29 http://archive.ubuntu.com/ubuntu jammy-updates/multiverse amd64 Packages [46.5 kB]                                                                                                      
Get:30 http://archive.ubuntu.com/ubuntu jammy-updates/multiverse Translation-en [11.8 kB]                                                                                                       
Get:31 http://archive.ubuntu.com/ubuntu jammy-updates/multiverse amd64 c-n-f Metadata [592 B]
Get:32 http://archive.ubuntu.com/ubuntu jammy-backports/main amd64 Packages [68.8 kB]
Get:33 http://archive.ubuntu.com/ubuntu jammy-backports/main Translation-en [11.4 kB]                                                                                                           
Get:34 http://archive.ubuntu.com/ubuntu jammy-backports/main amd64 c-n-f Metadata [392 B]                                                                                                       
Get:35 http://archive.ubuntu.com/ubuntu jammy-backports/universe amd64 Packages [30.0 kB]
Get:36 http://archive.ubuntu.com/ubuntu jammy-backports/universe Translation-en [16.5 kB]                                                                                                       
Get:37 http://archive.ubuntu.com/ubuntu jammy-backports/universe amd64 c-n-f Metadata [672 B]                                                                                                   
Fetched 17.9 MB in 7s (2737 kB/s)
Reading package lists... Done
```

아래 명령어를 입력하여 최신 패키지 목록을 받아오자.
```shell
# devbear@BOOK-MB2VJ96366:/mnt/c/Users/litl/PycharmProjects/gitProject/StudyDev$ 
sudo apt update
```
위 명령어를 입력하자, jammy 와 관련된 정보들을 읽는 것이 보였다.
```shell
## 결과 
Hit:1 http://archive.ubuntu.com/ubuntu jammy InRelease                                                                                                               
Hit:2 http://security.ubuntu.com/ubuntu jammy-security InRelease                 
Hit:3 http://archive.ubuntu.com/ubuntu jammy-updates InRelease                   
Hit:4 http://archive.ubuntu.com/ubuntu jammy-backports InRelease
Hit:5 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy InRelease
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
147 packages can be upgraded. Run 'apt list --upgradable' to see them.
```
sudo apt install python3.12 python3.12-venv python3.12-dev -y

파이썬 3.12 버전을 설치하자. (Ubuntu 의 환경을 바꾸는 것이 아닌, 윈도우에서 설치파일 깔 듯 설치 하는 것)
```shell
sudo apt install python3.12 python3.12-venv python3.12-dev -y
```
```shell
# 결과
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
The following additional packages will be installed:
  libpython3.12 libpython3.12-dev libpython3.12-stdlib mailcap mime-support
The following NEW packages will be installed:
  libpython3.12 libpython3.12-dev libpython3.12-stdlib mailcap mime-support python3.12 python3.12-dev python3.12-venv
0 upgraded, 8 newly installed, 0 to remove and 147 not upgraded.
Need to get 15.8 MB of archives.
After this operation, 62.8 MB of additional disk space will be used.
Get:1 http://archive.ubuntu.com/ubuntu jammy/main amd64 mailcap all 3.70+nmu1ubuntu1 [23.8 kB]
Get:2 http://archive.ubuntu.com/ubuntu jammy/main amd64 mime-support all 3.66 [3696 B]   
Get:3 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy/main amd64 libpython3.12-stdlib amd64 3.12.11-1+jammy1 [2870 kB]
Get:4 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy/main amd64 libpython3.12 amd64 3.12.11-1+jammy1 [2372 kB]                                                                   
Get:5 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy/main amd64 libpython3.12-dev amd64 3.12.11-1+jammy1 [5692 kB]                                                                
Get:6 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy/main amd64 python3.12 amd64 3.12.11-1+jammy1 [2517 kB]                                                                       
Get:7 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy/main amd64 python3.12-dev amd64 3.12.11-1+jammy1 [498 kB]                                                                    
Get:8 https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy/main amd64 python3.12-venv amd64 3.12.11-1+jammy1 [1796 kB]                                                                  
Fetched 15.8 MB in 1min 1s (257 kB/s)                                                                                                                                                           
Selecting previously unselected package mailcap.
(Reading database ... 31411 files and directories currently installed.)                                                                                                                         
Preparing to unpack .../0-mailcap_3.70+nmu1ubuntu1_all.deb ...                                                                                                                                  
Unpacking mailcap (3.70+nmu1ubuntu1) ...                                                                                                                                                        
Selecting previously unselected package mime-support.                                                                                                                                           
Preparing to unpack .../1-mime-support_3.66_all.deb ...                                                                                                                                         
Unpacking mime-support (3.66) ...                                                                                                                                                               
Selecting previously unselected package libpython3.12-stdlib:amd64.                                                                                                                             
Preparing to unpack .../2-libpython3.12-stdlib_3.12.11-1+jammy1_amd64.deb ...                                                                                                                   
Unpacking libpython3.12-stdlib:amd64 (3.12.11-1+jammy1) ...                                                                                                                                     
Selecting previously unselected package libpython3.12:amd64.                                                                                                                                    
Preparing to unpack .../3-libpython3.12_3.12.11-1+jammy1_amd64.deb ...                                                                                                                          
Unpacking libpython3.12:amd64 (3.12.11-1+jammy1) ...                                                                                                                                            
Selecting previously unselected package libpython3.12-dev:amd64.                                                                                                                                
Preparing to unpack .../4-libpython3.12-dev_3.12.11-1+jammy1_amd64.deb ...                                                                                                                      
Unpacking libpython3.12-dev:amd64 (3.12.11-1+jammy1) ...                                                                                                                                        
Selecting previously unselected package python3.12.                                                                                                                                             
Preparing to unpack .../5-python3.12_3.12.11-1+jammy1_amd64.deb ...                                                                                                                             
Unpacking python3.12 (3.12.11-1+jammy1) ...                                                                                                                                                     
Selecting previously unselected package python3.12-dev.                                                                                                                                         
Preparing to unpack .../6-python3.12-dev_3.12.11-1+jammy1_amd64.deb ...                                                                                                                         
Unpacking python3.12-dev (3.12.11-1+jammy1) ...                                                                                                                                                 
Selecting previously unselected package python3.12-venv.                                                                                                                                        
Preparing to unpack .../7-python3.12-venv_3.12.11-1+jammy1_amd64.deb ...                                                                                                                        
Unpacking python3.12-venv (3.12.11-1+jammy1) ...                                                                                                                                                
Setting up mailcap (3.70+nmu1ubuntu1) ...                                                                                                                                                       
Setting up mime-support (3.66) ...                                                                                                                                                              
Setting up libpython3.12-stdlib:amd64 (3.12.11-1+jammy1) ...                                                                                                                                    
Setting up python3.12 (3.12.11-1+jammy1) ...                                                                                                                                                    
Setting up libpython3.12:amd64 (3.12.11-1+jammy1) ...                                                                                                                                           
Setting up python3.12-venv (3.12.11-1+jammy1) ...                                                                                                                                               
Setting up libpython3.12-dev:amd64 (3.12.11-1+jammy1) ...                                                                                                                                       
Setting up python3.12-dev (3.12.11-1+jammy1) ...                                                                                                                                                
Processing triggers for man-db (2.10.2-1) ...   

```
설치가 되었으니, 가상환경을 생성하자.
```shell
devbear@BOOK-MB2VJ96366:/mnt/c/Users/litl/PycharmProjects/gitProject/StudyDev$ cd FY2025LLM                                                                                                     
devbear@BOOK-MB2VJ96366:/mnt/c/Users/litl/PycharmProjects/gitProject/StudyDev/FY2025LLM$  
python3.12 -m venv venvLinuxPy312
```
가상환경이 생성되었으니, 가상환경을 활성화 하자.
```shell
source venvLinuxPy312/bin/activate
```

```shell
export PYTHONPATH=/mnt/c/Users/litl/PycharmProjects/gitProject/StudyDev/FY2025LLM
python3 Elasticsearch/testrun.py
```

우분투환경에서 가상환경을 만들고, elasticsearch 서버를 연결하려 하자 connect False 가 떴다.
WSL에서 localhost는 WSL 리눅스 내부를 의미하고,
Windows에서 열어놓은 localhost와는 완전히 별개 여서 그런 것 같다.

##### Windows host IP 확인하기
```shell
((venvLinuxPy312) ) devbear@BOOK-MB2VJ96366:/mnt/c/Users/litl/PycharmProjects/gitProject/StudyDev/FY2025LLM$ cat /etc/resolv.conf | grep nameserver
```
```shell
# 결과
nameserver 10.255.255.254
```
```shell

```
curl -k https://10.255.255.254:9200
```shell
# 결과
curl: (7) Failed to connect to 10.255.255.254 port 9200 after 0 ms: Connection refused
```
Window 에서 elasticsearch 서버 설정을 변경해줘야 할것같다. (접근 허용하도록)

---

일단, VSCode 로 옮겨 왔다. (아무래도 Pycharm 무료 버전이 WSL 호환이 안되는 듯 하다.)  
Connect to WSL 에서 Ubuntu 환경을 클릭하고,  
/home/devbear/dev_projects/StudyDev/FY2025LLM 을 선택했다.  

그리고 다시 가상환경을 만들어주었다.
```shell
# 우분투 버전 조회 하기
# devbear@BOOK-MB2VJ96366:~/dev_projects/StudyDev/FY2025LLM$ 
lsb_release -a
# 가상 환경 생성하기
python3.12 -m venv venvUbun2204Py312
# 가상 환경 활성화 하기
devbear@BOOK-MB2VJ96366:~/dev_projects/StudyDev/FY2025LLM$ source venvUbun2204Py312/bin/activate

```

```shell
# 아래 명령어는 시스템 환경에서 실행
sudo apt update
sudo apt install -y libgsf-1-dev libxml2-dev libxslt1-dev python3-dev
```
