""" hashlib 을 사용한 암호화 처리 """

""" 비밀번호 암호화 관련 보안 규정

### KISA(한국인터넷진흥원) 권장사항

- 비밀번호는 SHA-256 이상을 사용하여 해시 저장
- 솔트(Salt) 추가 필수
- 64바이트(512비트) 이상의 해시 값 저장 권장
- PBKDF2, bcrypt, Argon2 등의 키 스트레칭(key stretching) 기법 활용 권장

### NIST(미국 국립표준기술연구소, NIST 800-63B 가이드라인)

- 최소 32바이트(256비트) 이상의 해시 적용 권장
- SHA-256, bcrypt, scrypt, Argon2 등의 알고리즘 권장
- 저장 시 솔트를 추가해야 함
- 8자 이상의 길이 제한(단, 실제 서비스에서는 최소 12~16자 이상 권장)

"""

"""
40 바이트 또는 44 바이트 비밀번호 해싱 이란?

보안 규정에서, 40바이트, 44바이트를 사용하는 것은 일반적으로 Base64 인코딩된 해시 값 때문이다.
SHA-256의 경우 해시 값은 32바이트(256 비트) 이나, 
이를 Base64로 인코딩하면 44바이트 길이가 된다.

"""

# Step 1. SHA-256을 사용하여 44바이트 문자열 만들기 (Base64 인코딩)
import hashlib
import base64

def hash_password_44(password: str) -> str:
    hashed = hashlib.sha256(password.encode()).digest() # 32바이트의 해시 생성
    return base64.b64encode(hashed).decode() # 44바이트로 Base64 인코딩

pw = "mypassword"
text_sha256_44byte = hash_password_44(pw) # SHA-256 -> 44바이트

print("SHA-256 해시 44byte 결과: ", text_sha256_44byte)

# Step 2. SHA-1을 사용하여 20바이트(160비트) 해시를 생성하고 16진수hex로 변환하여 40바이트 문자열 만들기
def hash_password_40(password: str) -> str:
    return hashlib.sha1(password.encode()).hexdigest() # 40 바이트

text_sha1_40byte = hash_password_40(pw)
print("SHA-1 해시 40byte 결과: ", text_sha1_40byte)
# 주의: SHA-1은 보안성이 약하여 금융 및 기업 보안에서는 사용을 권장하지 않는다고 한다.
# SHA-256 또는 bcrypt, PBKDF2, Argon2 사용을 권장한다.

# Step 3. PBKDF2 로 40 혹은 44 byte 만들기
def hash_password_pbkdf2(password: str, salt: bytes=b"random_salt") -> str:
    # 32 byte == 256 bit
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000, dklen=32)
    return base64.b64encode(key).decode() # 44 byte Base64 encoding

text_sha256_pbkdf2 = hash_password_pbkdf2(pw)
print("SHA-256 PDKDF2해시 44byte 결과: ", text_sha256_pbkdf2)


""" 정리하면
[1] 알고리즘 SHA-1 (hex)
길이: 40 byte
특징: 안전하지 않음, 사용 비추천

[2] 알고리즘 SHA-256 (hex)
길이: 64 byte
특징: 일반적인 해싱, 보안성이 강함

[3] 알고리즘 SHA-256 (Base64) 
길이: 44 byte
특징: 금융권, 기업에서 요구하는 포맷

[4] 알고리즘 PBKDF2 (Base64)
길이: 44 byte
특징: 반복 연산으로 보안 강화

[5] 알고리즘 bcrypt
길이: 60 byte
특징: 솔트 포함, 강력한 보안성
"""

""" salt 가 왜 나오는가??

위와 같이 단순히 암호화 하는 것은, 입력이 같으면 항상 같은 암호문자열을 생성하기 때문에
보안상 취약하다고 한다.

때문에 솔트를 처장하여, 해시를 다시 계산하고 비교한 다고 한다.

Salt 란?
비밀번호를 해싱할 때, 같은 비밀번호라도 항상 다른 해시값을 생성하기 위해 추가하는,
랜덤 데이터이다.

같은 비밀번호라도 서로 다른 해시값을 가지도록 만들기 위해 사용하는 것이다.
이는 Rainbow Table 공격을 방어하며,
사전 공경(Dictionary Attack) 공격을 방어하고
무차별 대입 공격 (Brute Force Attack) 난이도를 증가시킬 수 있다.

"""

""" Salt 를 사용한 비밀번호 검증 방식 흐름 

[1] 비밀번호 저장할 때

랜덤한 솔트(Salt) 생성
비밀번호 + 솔트를 이용해 해싱
"솔트:해시값" 형식으로 저장

[2] 비밀번호 검증할 때

저장된 "솔트:해시값" 데이터를 불러옴
: 기준으로 솔트와 해시를 분리
입력된 비밀번호를 가져온 솔트와 함께 다시 해싱
기존 해시값과 비교하여 로그인 여부 결정


"""

# 하지만 룰이 44바이트라면 salt는 무시하고, SHA-256 -> Base64 를 사용하면 된다!