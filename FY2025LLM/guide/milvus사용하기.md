# docker와 docker compose 설치 후

git clone https://github.com/milvus-io/milvus.git
cd milvus/deployments/docker/standalone

docker compose up -d


#### 확인하기
```bash 
docker compose ps 
```
`docker compose ps` 명령어는 현재 디렉토리에 있는 docker-compose.yml 파일을 기준으로 상태를 보여준다.  

```bash
# 결과
 PS C:\Users\litl\PycharmProjects\gitProject\StudyDev\FY2025LLM\milvus\milvus\deployments\docker\standalone> docker compose ps   
>> time="2025-09-13T12:04:50+09:00" level=warning msg="C:\\Users\\litl\\PycharmProjects\\gitProject\\StudyDev\\FY2025LLM\\milvus\\milvus\\deployments\\docker\\standalone\\docker-compose.yml: `version` is obsolete"
>> NAME                IMAGE                                      COMMAND                   SERVICE
>>       CREATED          STATUS                    PORTS
>> milvus-etcd         quay.io/coreos/etcd:v3.5.18                "etcd -advertise-cli…"   etcd 
        13 minutes ago   Up 13 minutes (healthy)   2379-2380/tcp
>> milvus-minio        minio/minio:RELEASE.2024-05-28T17-19-04Z   "/usr/bin/docker-ent…"   minio
        13 minutes ago   Up 13 minutes (healthy)   0.0.0.0:9000-9001->9000-9001/tcp
>> milvus-standalone   milvusdb/milvus:v2.6.1                     "/tini -- milvus run…"   standalone   13 minutes ago   Up 13 minutes (healthy)   0.0.0.0:9091->9091/tcp, 0.0.0.0:19530->19530/tcp
```
결과를 보면,  
milvus-etcd, milvus-minio, milvus-standalone 세 개의 컨테이너가 모두 Up (healthy) 상태,  즉 Milvus가 필요한 모든 컴포넌트를 정상 구동 완료한 것이 보인다.  

19530/tcp → gRPC 연결용 (SDK, Python 클라이언트에서 사용)  
9091/tcp → REST API 연결용 (HTTP 호출로 확인 가능)  

