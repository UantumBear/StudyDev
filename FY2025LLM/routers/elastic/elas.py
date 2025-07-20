
"""
@경로 routers/elas.py
@용도 elasticsearch의 테스트 용 라우터
"""


from fastapi import APIRouter
import requests
from requests.auth import HTTPBasicAuth
from config import conf
import subprocess


ELAS = APIRouter(prefix="/elas", tags=["Elasticsearch"])

@ELAS.get("/indices")
def get_indices():
    url = "https://localhost:9200/_cat/indices?v"
    try:
        response = requests.get(
            url,
            auth=HTTPBasicAuth(conf.ELASTICSEARCH_ID_LOC, conf.ELASTICSEARCH_PW_LOC),
            verify=False # 로컬 개발 시에만 사용 할 것
        )
        response.raise_for_status()

        # 텍스트 파싱
        lines = response.text.strip().split("\n")
        headers = lines[0].split()
        rows = [dict(zip(headers, line.split())) for line in lines[1:]]

        return {"indices": rows}

    except Exception as e:
        return {"error": str(e)}


@ELAS.get("/shards")
def get_shards():
    url = "https://localhost:9200/_cat/shards?v&bytes=mb"
    try:
        response = requests.get(
            url,
            auth=HTTPBasicAuth(conf.ELASTICSEARCH_ID_LOC, conf.ELASTICSEARCH_PW_LOC),
            verify=False  # 개발 환경
        )
        response.raise_for_status()

        # 텍스트 파싱 → JSON 배열로 변환
        lines = response.text.strip().split("\n")
        headers = lines[0].split()
        rows = [dict(zip(headers, line.split(None, len(headers) - 1))) for line in lines[1:]]
        return {"shards": rows}

    except Exception as e:
        return {"error": str(e)}

@ELAS.get("/disk-usage")
def get_elasticsearch_disk_usage():
    try:
        powershell_cmd = [
            "powershell",
            "-Command",
            f'Get-ChildItem "{conf.ELSETICSEARCH_SERVER_DIR}" -Recurse | Measure-Object -Property Length -Sum'
        ]

        result = subprocess.run(
            powershell_cmd,
            capture_output=True,
            text=True,
            check=True
        )

        return {"output": result.stdout.strip()}

    except subprocess.CalledProcessError as e:
        return {"error": e.stderr.strip()}
    except Exception as e:
        return {"error": str(e)}

