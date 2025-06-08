##### 25.06.07
## OpenMP

##### OpenMP 란?  
병렬 프로그래밍을 쉽게 구현하기 위한  C, C++, Fortran 용 API  
컴파일러 지시문, 라이브러리 함수, 환경 변수로 구성된다.  
간단한 문법으로 병렬화를 가능하게 한다.

```c++
#pragma omp parallel
{
  // 병렬 실행할 코드
  
}
```
위와 같은 소스로, 여러 스레드를 동시에 실행한다.  
```c++
#pragma omp parallel() : 병렬 영역을 생성, 데이터 분배 방식에는 Cyclic 과 Block 이 있다.
omp_get_thread_num()   : 스레드 ID 
opm_set_num_threads(n) : 사용할 스레드 수 지정
```

OpenMP는 모든 스레드가 전역 변수를 공유한다.  
공유 변수 사용 시 Race Condition 이 발생할 수 있다.  
* Race Condition 이란?
* Race Condition 해결 방법
```c++
#pragma omp critical(): 한 번에 하나의 스레드만 실행
#pragma omp atomic()  : 변수 하나에 대한 원자적 연산   
```

##### 병렬 루프
```c++
#pragma omp parallel for 
for (int i=0; i<N; i++){
    // 병렬 반복
    // 반복 루프를 스레드끼리 자동으로 분할하여 수행한다.
    // reduction(op:var)를 통해 병렬 루프 내 변수 합치기가 가능하다.
}
```

##### 주요 동기화 방법
```c++
#pragma omp barrier() : 모든 스레드가 도달할 때까지 대기
#pragma omp master()  : 마스터 스레드만 실행
#pragma omp single()  : 하나의 스레드만 실행 (끝나면 barrier 발생)
omp_lock_t()          : 수동 잠금으로 임계 구역을 보호한다.
```

##### 병렬 PI 계산
###### 순차 코드
```c++
for (i=0; i<N; i++){
    x=(i+0.5)*step;
    sum+=4.0/(1+x*x);
}
```
###### 병렬 코드 (Reduction 사용)
```c++
#pragma imp parallel for reduction(+:sum)
for (i=0; i<N; i++){
    x=(i+0.5)*step;
    sum+=4.0/(1+x*x);
}
```

##### Loop Schedule Clause 스케줄 설정
루프 반복을 어떻게 스레드에 나눌지 지정한다.
```c++
schedule(static, chunk)  : 고정 간격으로 분배
schedule(dynamic, chunk) : 작업 큐에서 동적으로 가져감 
schedule(guided)         : 큰 chunk로 시작해 점점 작아짐.
schedule(runtime)        : 실행 시 환경 변수로 설정
```

##### 데이터 공유 속성
```c++
private(x)      : 각 스레드가 독립된 x 변수를 가진다.
firstprivate(x) : 각 스레드가 초기값을 복사
lastprivate(x)  : 마지막 반복의 값을 메인 변수에 복사
```

##### 핵심 개념
- 스레드: 병렬로 실행되는 코드 단위
- parallel region : #pragma omp parallel 로 정의된 병렬 코드 블록
- race condition : 여러 스레드가 동시에 변수에 접근해 발생하는 오류
- critical section : 한번에 하나의 스레드만 실행되는 코드 영역
- reduction : 병렬로 계산된 결과를 하나로 합치는 방식
- schedult : 반복문 작업을 스레드에 어떻게 분배할 지 결정