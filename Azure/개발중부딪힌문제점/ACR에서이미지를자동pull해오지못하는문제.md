상황:

- AWS (아마존 웹서비스) 에 Base Image 저장 (수동작업)
- Bitbucket 의 Dockerfile 이 AWS에서 이미지를 받아와서 Dockerfile 을 기반으로 이미지를 build 하여 ACR(애저 컨테이너 레지스트리)에 이미지 push 하는 CI/CD 파이프라인이 구축되어있음.

!! 그런데 ACR에 push 된 이미지가 ...Azure WS(애저 웹서비스)에서 자동 배포가 되지 않음..

container 재실행이 안되는게 문제인지,
pull 자체가 안되는게 문제인지
어느 단계에서 문제인지를 모르겠음.

확인 사항:
(1) Azure 웹 서비스 - 배포센터 에 webhooks URL 은 잘 등록되어 있음.
https://$app-kc-gpt-dev-app-admin:
AkdymYH1KysjoGjQE5X9xm6eT8vEkvdbodmpbvKTJkcomByRrrFyowMTgCko
@app-kc-gpt-dev-app-admin.scm.azurewebsites.net/api/registry/webhook
위 링크가 등록되어 있고,
컨테이너 레지스트리 웹후크 의 '구성'-서비스URL 에도 동일하게 들어가 있음.


알고보니 그냥 구축이 안되었었다고 한다..

