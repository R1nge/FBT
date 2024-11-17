class UDPClient:
  import socket
  
  msgFromClient       = "Hello UDP Server"
  bytesToSend         = str.encode(msgFromClient)
  serverAddressPort   = ("127.0.0.1", 9050)
  bufferSize          = 1024

  def sendHello(self):
    UDPClientSocket = self.socket.socket(family=self.socket.AF_INET, type=self.socket.SOCK_DGRAM)
    UDPClientSocket.sendto(self.bytesToSend, self.serverAddressPort)
    msgFromServer = UDPClientSocket.recvfrom(self.bufferSize)
    msg = "Message from Server {}".format(msgFromServer[0])
    print(msg)