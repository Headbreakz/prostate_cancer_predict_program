import socket
import numpy as np
import cv2
import pickle
import struct
import io
import time
import zlib

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


# 전립선 암 등급 통신
def aws_connect(file_path):
    # 서버의 주소입니다. hostname 또는 ip address를 사용할 수 있습니다.
    HOST= ''
    # 서버에서 지정해 놓은 포트 번호입니다.
    PORT= 8895

    img = cv2.imread(file_path)
    print("사진크기:{}".format(img.shape))
    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
    img = cv2.imencode('.png', img, encode_param)
    data = pickle.dumps(img, protocol=3)
    # data = pickle.dumps(img, 0)
    size = len(data)
    # print(size)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection = client_socket.makefile('wb')
    client_socket.connect((HOST, PORT))
    client_socket.sendall(struct.pack(">L", size) + data)
    # while True:
    #     client_socket.sendall(struct.pack(">L", size) + data)
    #     break
        # message = '1'
        # client_socket.send(message.encode())

        # length = recvall(client_socket,16)
        # stringData = recvall(client_socket, int(length))
        # data = np.frombuffer(stringData, dtype='uint8')

        # decimg=cv2.imdecode(data,1)
        # #cv2.imshow('Image',decimg)

        # key = cv2.waitKey(1)
        # if key == 27:
        #     break
    data = client_socket.recv(1024)

    print("ISUP 점수 값 : {}".format(data.decode()))
    print('close socket')
    client_socket.close()

    return data.decode()

#전립선 암 부위 예측 통신
def aws_connect2(file_path):
    # 서버의 주소입니다. hostname 또는 ip address를 사용할 수 있습니다.
    HOST= ''
    # 서버에서 지정해 놓은 포트 번호입니다.
    PORT= 8896

    img = cv2.imread(file_path)
    print("사진크기:{}".format(img.shape))
    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
    img = cv2.imencode('.png', img, encode_param)
    data = pickle.dumps(img, protocol=3)
    # data = pickle.dumps(img, 0)
    size = len(data)
    # print(size)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection = client_socket.makefile('wb')
    client_socket.connect((HOST, PORT))
    client_socket.sendall(struct.pack(">L", size) + data)
    # while True:
    #     client_socket.sendall(struct.pack(">L", size) + data)
    #     break
        # message = '1'
        # client_socket.send(message.encode())

        # length = recvall(client_socket,16)
        # stringData = recvall(client_socket, int(length))
        # data = np.frombuffer(stringData, dtype='uint8')

        # decimg=cv2.imdecode(data,1)
        # #cv2.imshow('Image',decimg)

        # key = cv2.waitKey(1)
        # if key == 27:
        #     break

    # data = client_socket.recv(1024)
    data = b""
    payload_size = struct.calcsize(">L")
    print("payload_size: {}".format(payload_size))
    while True:
        while len(data) < payload_size:
            print("Recv: {}".format(len(data)))
            data += client_socket.recv(4096)

        print("Done Recv: {}".format(len(data)))
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        print("msg_size: {}".format(msg_size))
        while len(data) < msg_size:
            data += client_socket.recv(4096)
            #print(len(data))
        frame_data = data[:msg_size]
        data = data[msg_size:]
        

        # pick_data=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        pick_data = pickle.loads(frame_data)
        
        # print(frame)
        # print(np.load(image))
        print(pick_data)
        print(len(pick_data))
        image = cv2.imdecode(pick_data[1], cv2.IMREAD_COLOR)
        print(type(image))
        print(image.shape)

        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        cv2.imwrite("output1.png", image)

        break

    # print("ISUP 점수 값 : {}".format(data.decode()))
    # cv2.imwrite("output1.png", image)
    print('close socket')
    client_socket.close()

