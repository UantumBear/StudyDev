##### Step 1. Azure Open AI Resource 만들기
![azureOpanAI리소스만들기.png](./datas/azureOpanAI리소스만들기.png)

![azureOpanAI리소스만들기2.png](./datas/azureOpanAI리소스만들기2.png)

위와 같이 리소스 그룹을 우선적으로 만들어주어야 한다.  
리소스 그룹 이름은 그냥 설정하면 되며,  
인스턴스 세부 정보 - 이름 은 전세계적으로 고유해야 한다.  

현재 학생 구독을 인증받아 student-resource-group 으로 이름을 정했는데,  
추후 리소스 그룹 간 리소스 이동이 가능하다고 하므로, 고민말고 설정해도 될 듯 하다.  

![azureOpanAI리소스만들기3.png](./datas/azureOpanAI리소스만들기3.png)
그 다음은 네트워크를 설정한다. api key 가 유출되면 안되므로 selected 를 권장하나,  
카페에서 주로 작업하므로 나는 전체 인터넷으로 선택하였다.

![azureOpanAI리소스만들기4.png](./datas/azureOpanAI리소스만들기4.png)
이제 태그를 선택한다.   
나는 student 리소스 구분을 위해 student 태그와
내 닉네임인 uantumbear 를 사용하였다.

다음 단계인 검토+ 제출을 완료하자,  
래와 같은 일정 시간 소요 후, 배포 완료 화면이 나타났다.
![azureOpanAI리소스만들기5.png](./datas/azureOpanAI리소스만들기5.png)
배포 이름은 Azure에서 자동으로 생성한 리소스 이름,  
상관관계ID는 Azure에서 작업 추적을 위해 사용하는 ID 라고 한다.

---

이제 리소스로 이동해보자.  
아래와 같은 화면에서 개발에 필요한 endpoint 와 key 를 볼 수 있다.  
절대 외부로 노출되지 않도록 관리하자.
![azureOpanAI리소스1.png](./datas/azureOpanAI리소스1.png)


---

Azure Open AI 모델 사용을 위해서는 모델 배포를 해야한다.
먼저 Azure AI Foundry 포털에 로그인을 한다.
https://ai.azure.com/
![azureOpanAI리소스1.png](./datas/azureAIFoundry1.png)

우측 메뉴를 보면, 진행 단계를 확인할 수 있다.  
현재 Azure 구독은 만들었고, 프로젝트 만들기를 해야한다.

만들기를 클릭하면, 프로젝트 이름과 세부 설정이 자동으로 생성되어있다.  
'사용자지정'을 클릭하여, 세부 설정을 변경할 수 있다.

**프로젝트이름**은 언제든지 변경 가능하며,  
구독은 아까 생성했던 student 구독을 사용하면 된다. (해당 창에서는 변경 불가)  
**허브**는 AI 프로젝트 내에서 리소스를 연결하고 조율하는 중심 역할을 하는 서비스라고 한다. 해당 창에서 변경 가능하다.

아래와 같이, 자동 생성되어있던 리소스 그룹과 openAI 리소스 선택창을   
아까 생성했던 **리소스 그룹**과 **openAI 리소스**로 변경해주었다.    
![azureAIFoundry2.png](./datas/azureAIFoundry2.png)

아래와 같이 리소스를 만들고 있다. 꽤 시간 소요된다.
![azureAIFoundry3.png](./datas/azureAIFoundry3.png)

배포가 완료되자 모델 배포 탐색이 가능했다.
![azureAIFoundry4.png](./datas/azureAIFoundry4.png)
위와 같이 정말 다양한 모델들이 있다.
일단 우측에서 추천하는 gpt-4o-mini 로 배포해보고자 한다.

![azureAIFoundry5.png](./datas/azureAIFoundry5.png)
위와 같이 할당량 부족 문제가 떴다.
Azure에는 리전별 서비스 제공 정책이 있다고 하는데, korean central 에는 해당 할당량이 없는 듯 하다.
다른 지역으로 변경하자.
챗 gpt는 East US 를 추천하므로 해당 리전으로 변경하겠다.
맙소사. 리소스 그룹의 리전과 모델 리전은 달라고 되지만 비용 절감을 위해서는 리전을 맞추는걸 추천한다고 한다.

새로운 리소스 그룹을 만들고 리전을 변경하겠다.