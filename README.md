# SOCKET_FL
***Federated Learning based on Socket TCP***

각 서버와 클라이언트 로컬 환경에 SocketFL_server 내의 파일들과 SocketFL_client 내의 파일들을 담으시고

server의 가상환경에서는 

```python socket_server.py``` or ```python3 socket_server.py```

client의 가상환경에서는 

```python learning_client.py``` or ```python3 learning_client.py``` 를 입력해주세요.

---
***Requirement***

- python 3.x 이상

- pytorch or torch (본인의 python 버전과 호환되는 버전)

- conda 

---
***Warning***

코드 내의 HOST의 IP주소는 server의 외부 IP주소를 입력해주시고 

PORT는 상황에 맞게 변경하시면 됩니다. (단, 공유기 사용 시 포트포워딩 필수)

AWS의 경우에는 가동할 때마다 외부 IP 주소가 변하기 때문에 주의해주시기 바랍니다.
