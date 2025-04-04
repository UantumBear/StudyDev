"""
(1) 비밀번호는 단방향 암호화 하야 DB에 저장
(2) 관리자 웹에서 관리자도 암호화된 회원의 password 를 볼 수 없도록 함
(3) 최소 8~12자리 이상, 대문자+소문자+숫자+특수문자 조합 필수
(4) 3~6개월마다 변경하도록 정책 설정
(5) 이전 비밀번호 재사용 금지

"""

# Step 1. 비밀번호가 8자리 이상이며, 대문자+소문자+숫자+특수문자를 포함하는 지 검증
import re
def validate_password(password: str) -> bool:
    """비밀번호가 8자리 이상이며, 대문자, 소문자, 숫자, 특수문자를 포함하는지 검증"""
    #pattern = r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[\W_]).{8,}$'
    pattern = r'^(?!.*\s)(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[!@#$%^&*()]).{8,}$'
    """비밀번호가 8자리 이상이며, 대문자, 소문자, 숫자, 특수문자(!@#$%^&*()만 허용), 공백금지 를 포함하는지 검증"""
    return bool(re.match(pattern, password))

print(validate_password("Yjy0118!"))

"""
정규패턴식
^            : 문자열의 시작을 의미한다. 즉 입력값 전체가 검증 대상이다.
(?=       )  : 전방 탐색(Lookahead) , 앞으로의 문자열이 특정 조건을 만족하는지 확인한다.
.*           : 임의의 문자를 포함하여 0개 이상 찾아라.
(?!.*\s)     : 공백 금지
(?=.*[A-Z])  : 대문자를 1개 이상 포함하는 지 찾아라.
(?=.*[a-z])  : 소문자를 1개 이상 포함하는 지 찾아라.
(?=.*\d)     : 숫자를 1개 이상 포함하는 지 찾아라.
(?=.*[\W_])  : 특수문자(\W) 또는 _를 1개 이상 포함
(?=.*[!@#$%^&*()]) : 특수만자 !@#$%^&*() 중 최소 1개 포함
.{8,}        : 전체 문자열 길이가 최소 8자 이상
$            : 전체 문자열의 끝을 의미한다.
"""

"""
정규표현식은 자바스크립트에서도 똑같이 사용 가능하니 알아두자.
function validatePassword(pw){
    const pattern = /^(?!.*\s)(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[!@#$%^&*()]).{8,}$/;
    return pattern.test(pw);
}
console.log(validatePassword("Yjy0118!"));

프론트엔드를 우회한 비정상적인 요청을 방지하기 위해 Front+Back 에서 모두 처리한다.
"""