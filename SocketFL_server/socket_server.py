import socket
import sys
import pickle
import datetime
from _thread import *
from train_module import *


train_dataset = torch.load('./data_tensor/train_dataset')
dev_dataset = torch.load('./data_tensor/dev_dataset')

train_dataset, dev_dataset = random_split(train_dataset, [int(len(train_dataset) * 0.83), int(len(train_dataset) * 0.17)])


# Variable Set
client_sockets = []
max_client = 5
client_count = 0
current_round = 1
learn_flag = False

HOST = '168.131.154.188'
PORT = 8080

total_train_size = 49800
fraction = (total_train_size / max_client) / total_train_size

# server socket create
print(">> Server Start ")
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen()

global_net = to_device(FederatedNet(), device)
history = []
curr_parameters = global_net.get_parameters()
new_parameters = dict([(layer_name, {'weight': 0, 'bias': 0}) for layer_name in curr_parameters])


def encodeParams(parameter) :
    return pickle.dumps(parameter, -1)

def decodeParams(parameter) :
    return pickle.loads(parameter)

def threaded(client_socket, addr) :
    global client_count
    global learn_flag
    global global_net
    global curr_parameters
    global new_parameters

    current_parameter = curr_parameters
    buffer = b''

    print("[{}] || Connected by : <{}|{}>".format(datetime.datetime.now(), addr[0], addr[1]))

    client_socket.send(encodeParams(current_parameter))

    print("[{}] >> Send Init parameters to <{}|{}>".format(datetime.datetime.now(), addr[0],  addr[1]))

    while True:
        try:   
            if not learn_flag :
                data = client_socket.recv(300000)
                buffer += data
                if (sys.getsizeof(buffer) > 265000):  

                    print("[{}] << Recive local parameters from <{}|{}> | size : {}".format(datetime.datetime.now(), addr[0],  addr[1], sys.getsizeof(buffer)))
                    client_parameters = decodeParams(buffer)
                    for layer_name in client_parameters:
                        new_parameters[layer_name]['weight'] += fraction * client_parameters[layer_name]['weight']
                        new_parameters[layer_name]['bias'] += fraction * client_parameters[layer_name]['bias']
                    print("[{}:{}]의 파라미터 적용 완료".format(addr[0],  addr[1]))
                    client_count += 1
                    buffer = b''

        except ConnectionResetError as e:
            print("[{}] | Disconnected by : <{}|{}>".format(datetime.datetime.now(), addr[0], addr[1]))
            break

    print(">>>>>> Thread 종료!")
    client_socket.close()

def globalLearning() :
    global client_count
    global max_client
    global current_round
    global learn_flag
    global client_sockets
    global curr_parameters
    global new_parameters

    while True:
        if(client_count >= max_client) :                       
            learn_flag = True
            print("[{}] || Round {} : Local parameters all selected".format(datetime.datetime.now(), current_round))

            
            global_net.apply_parameters(new_parameters) 
            train_loss, train_acc = global_net.evaluate(train_dataset)
            dev_loss, dev_acc = global_net.evaluate(dev_dataset)
            print('After round {}, train_loss = {}, train_acc = {}, dev_loss = {}, dev_acc = {}\n'.format(current_round, round(train_loss, 4), round(train_acc, 4), round(dev_loss, 4), round(dev_acc, 4)))
            history.append((train_loss, dev_loss))

            print("[{}] || Global train done.".format(datetime.datetime.now()))

            curr_parameters = global_net.get_parameters()
            new_parameters = dict([(layer_name, {'weight': 0, 'bias': 0}) for layer_name in curr_parameters])

            client_count = 0
            current_round += 1
            learn_flag = False

            if (current_round > 5) :
                print(">> 최종 라운드 학습 완료, 탈출하겠습니다.")
                break
            else :
                for client in client_sockets :
                    client.send(encodeParams(curr_parameters))

        
    
    print(">>>>>> Thread 종료!")

# Wait for connect all clients
try:
    
    start_new_thread(globalLearning,())
    while (client_count < max_client):
        print('>> Wait')

        client_socket, addr = server_socket.accept()
        client_sockets.append(client_socket)
        start_new_thread(threaded, (client_socket, addr))
        
except Exception as e :
    print ('[chobisang] ERROR : ',e)

finally:

    server_socket.close()
