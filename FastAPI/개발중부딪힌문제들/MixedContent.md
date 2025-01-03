##### 상황:

외부망 azure web server 에 띄워놓은 웹에서
특정 화면 (2차 팝업 등..) 에서만 Mixed Content 에러가 났다.


##### 문제 해결 방법:



##### 공부:
#### Mixed Content 오류란?
브라우저가 HTTPS로 제공되는 페이지에서 HTTP로 제공되는 리소스를 로드하려고 할 때 발생한다.
페이지는 HTTPS 로 제공되더라도, CSS, js 가 HTTP를 로드하면 해당 에러가 발생한다.


#### HTTP와 HTTPS
HPPTS 는 데이터를 암호화해서 보안을 유지하지만,
(HTTP Secure)
HTTP  는 데이터를 암호화하지 않는다는 차이가 있다.
(Hypertext Transfer Protocol)
