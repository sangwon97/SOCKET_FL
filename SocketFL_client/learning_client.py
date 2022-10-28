import socket
import pickle
import time
import uuid
import datetime
import matplotlib.pyplot as plt
from _thread import *
from train_module import *

# Variable Set
HOST = '168.131.154.188'
PORT = 8080

current_round = 1

# client socket create
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

client_dataset = torch.load('./data_tensor/client_dataset_01')
client = Client('client_01', client_dataset)  

def encodeParams(parameter) :
    return pickle.dumps(parameter)

def decodeParamse(parameter) :
    return pickle.loads(parameter)

def roundLearning(data) :
    global current_round

    global_params = decodeParamse(data)

    print("[{}] << Recive init parameters : {}".format(datetime.datetime.now(), data.decode()))
    print("[{}] || Round {} train start".format(datetime.datetime.now(), current_round))

    client_parameters = client.train(global_params)

    send_parameters = encodeParams(client_parameters)

    print("[{}] || Train done ".format(datetime.datetime.now()))

    client_socket.send(send_parameters)
    print("[{}] >> Send trained parameters : {}".format(datetime.datetime.now(), str(client_parameters.keys())))
    current_round += 1

while True:
    data = client_socket.recv(524288)

    if(data):
        roundLearning(data)

    if(current_round > 3):
        print(">> 최종 라운드 학습 완료, 탈출하겠습니다.")
        break

client_socket.close()