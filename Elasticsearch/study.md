### aggs 와 search

**search 란?**

문서를 직접 검색해서 가져오는 방식  
- query 로 조건을 주고  
- from, size 로 페이징 가능하다.  
- _source 로 원하는 필드만 뽑을 수 있다.  

**aggs 란? (aggregtion)**

통계, 그룹핑, 평균, 합계 같은 요약/집계 작업에 사용된다. "필드별로 집계하거나 그룹핑해줘!"  
- 문서를 직접 반환하지 않는다. (size:0)  
- terms, avg, sum, top_htis 등 사용 (top_hits를 써야 실제 문서를 가져올 수 있다.)  


### 파일 형식
#### HWP 파일 파싱 방법
구버전 .hwp 는 내부적으로 OLE2 구조를 사용한다.  
olefile 라이브러리를 사용한다. 
```shell
(venv312) PS C:\Users\litl\PycharmProjects\gitProject\StudyDev> 
pip install olefile 
```
pyhwp 라는 라이브러리를 받았는데, 이 라이브러리는 linux 기반이라고 한다.
설치를 했는데도 인식을 하지 못하는 것이 이 이유인 듯 하다.
