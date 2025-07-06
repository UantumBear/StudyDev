이곳은 챗봇을 구현하기 위해 쓰인 data 들을 관리하는 경로이다.  

카테고리별로 데이터를 관리하며,
data/ 바로 밑의 경로로 원본 데이터들을 관리한다.  


소스에 의해 전처리된 데이터는 전부 converted/ 하위에서 관리한다.

###### data/
| 경로       |          | 설명 | 데이터 사용 여부  |
|-----------|----------|----|------------------|
| CarrotAI  |          |    | T                |
| converted |          |    |                  |
|           | CarrotAI |    |                  |
| DevBear   |          |    | O                |
| groupware |          |    |                  |
| test      |          |    |                  |

※ T 는 개발을 위한 테스트에 이용

###### models/llama3.2-1B-hf/finetuned
| 경로       | 설명                           |
|----------|------------------------------|
| model_v1 |                              |
| model_v2 |                              |
| model_v3 |                              |
| model_v4 | CarrotAI + DevBear data 파인튜닝 |
|          |                              |
|          |                              |