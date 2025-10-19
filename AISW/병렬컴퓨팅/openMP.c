#include <stdio.h>
#include <omp.h> // OpenMP 기능을 사용하기 위한 헤더파일

int main() {

    // Test Function 1.  ... 문장 출력을 병렬로 실행해보기! 
    #pragma omp parallel  // 이 블록을 병렬로 실행하라
    {
        int id = omp_get_thread_num();
        printf("Hello from thread %d\n", id);
    }
    /* 결과 예시  :: 병렬 실행이기 때문에 이런식으로 찍힌 것.
        Hello from thread 2
        Hello from thread 20
        Hello from thread 13
        Hello from thread 11
        Hello from thread 7
        Hello from thread 9
        Hello from thread 16
        Hello from thread 15
        Hello from thread 19
        Hello from thread 8
        Hello from thread 10
        Hello from thread 12
        Hello from thread 1
        Hello from thread 3
        Hello from thread 17
        Hello from thread 14
        Hello from thread 5
        Hello from thread 0
        Hello from thread 21
        Hello from thread 6
        Hello from thread 18
        Hello from thread 4
        */

    // Test Function 2. ... 문장 출력을 병렬로 수행하보면서, 해당 쓰레드의 ID 값 출력해보기..!
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int total = omp_get_num_threads();
        printf("Thread ID: %d , Total Thread 수: %d\n", id, total);
    }

    
    // Test Function 3.  ... 병렬로 계산해보기...!
    long num_steps = 100000000; // 정밀도 (높을수록 느림)
    double step = 1.0 / (double) num_steps;
    double pi_serial = 0.0, pi_parallel = 0.0;
    double start_time, end_time;

    // 1. 순차 실행
    double sum_serial = 0.0;
    start_time = omp_get_wtime();
    for (long i = 0; i < num_steps; i++) {
        double x = (i + 0.5) * step;
        sum_serial += 4.0 / (1.0 + x * x);
    }
    pi_serial = step * sum_serial;
    end_time = omp_get_wtime();
    printf("[순차 실행] Pi = %.15f, 시간 = %.4f초\n", pi_serial, end_time - start_time);

    // 2. 병렬 실행
    double sum_parallel = 0.0;
    start_time = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum_parallel)
    for (long i = 0; i < num_steps; i++) {
        double x = (i + 0.5) * step;
        sum_parallel += 4.0 / (1.0 + x * x);
    }
    pi_parallel = step * sum_parallel;
    end_time = omp_get_wtime();
    printf("[병렬 실행] Pi = %.15f, 시간 = %.4f초\n", pi_parallel, end_time - start_time);
    /* 결과 예시:
        [순차 실행] Pi = 3.141592653590426, 시간 = 0.1876초
        [병렬 실행] Pi = 3.141592653589810, 시간 = 0.0341초
    */ 
    


    // gcc -fopenmp openMP.c -o openMP && ./openMP
    return 0;// main 함수 종료
}

