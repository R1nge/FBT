import zmq

class Server:
    def __init__(self, ip:str, port:str):
        self.ip = ip
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{self.ip}:{self.port}")
        
    def receive(self) -> bytes:
        message_rx = self.socket.recv()
        print(f"Socket has received a message: {message_rx}")
        return message_rx

    def send_string(self, string:str):
        print(f"Socket has sent a message: {string}")
        self.socket.send_string(string)     