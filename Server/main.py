from ai import AI
from server import Server
# from ai import AI
import asyncio

server = Server('127.0.0.1', '9085')

ai = AI()


async def main():
    request = server.receive()
    text_string = request.decode()
    server.send_string("hello")
    ai.init()
    ai.process()

    #while True:
        #ai.process()
        #await asyncio.sleep(0.1)

asyncio.run(main())
