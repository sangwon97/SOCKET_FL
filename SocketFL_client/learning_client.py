import socket
import pickle
import datetime
import sys
import time
from _thread import *
from train_module import *

# Variable Set
HOST = '168.131.154.188'
PORT = 8080

current_round = 1
choose_file = sys.argv[1]

if len(sys.argv) != 2:
    print("Insufficeint Argv")
    sys.exit()

# client socket create
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

client_dataset = torch.load('./data_tensor/client_dataset_0' + choose_file)
client = Client('client_01', client_dataset)  

buffer = b''

def encodeParams(parameter) :
    return pickle.dumps(parameter)

def decodeParams(parameter) :
    return pickle.loads(parameter)

def roundLearning(data) :
    global current_round
    global buffer

    global_params = decodeParams(data)
    start_time = time.time()

    # print("[{}] << Recive init parameters | size : {}".format(datetime.datetime.now(), sys.getsizeof(data)))
    # print("[{}] || Round {} train start".format(datetime.datetime.now(), current_round))

    client_parameters = client.train(global_params)
    
    # print("[{}] || Train done : {}".format(datetime.datetime.now(), len(encodeParams(client_parameters))))
    end_time = time.time()
    client_socket.send(encodeParams(client_parameters))
    print("[{}] >> Send trained parameters".format(datetime.datetime.now()))
    print(f"{end_time - start_time:.5f} sec")
    # print("[{}] >> Send trained parameters : {}".format(datetime.datetime.now(), str(client_parameters.keys())))
    print("----------------------------------------------")
    current_round += 1
    buffer = b''


while True:
    data = client_socket.recv(300000)
    buffer += data
    if (sys.getsizeof(buffer) > 265000):
        print("[{}] << Recive init parameters".format(datetime.datetime.now()))
        roundLearning(buffer)

    if(current_round > 5):
        time.sleep(10)
        print(">> 최종 라운드 학습 완료, 탈출하겠습니다.")
        break

client_socket.close()