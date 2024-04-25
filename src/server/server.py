# Code by Alexandre Haddad-Delaveau

from aiohttp import web
from threading import Thread
import asyncio

class HTTPServer:
    def __init__(self, listeners=[]):
        # Init variables
        self.pos_x = 0
        self.pos_y = 0
        self.pos_z = 0

        self.speed = 0
        self.rpm = 0
        self.gear = 0

        self.racing = False

        self.listeners = listeners

        # Add route
        app = web.Application()
        app.add_routes([web.post('/', self.handleRequest)])

        # Save Runner
        self.runner = web.AppRunner(app)

        # Start server
        Thread(target=self.start_server, daemon=True).start()

    def start_server(self):
        # Solution from https://stackoverflow.com/questions/51610074/how-to-run-an-aiohttp-server-in-a-thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.runner.setup())
        site = web.TCPSite(self.runner, 'localhost', 6969)
        loop.run_until_complete(site.start())
        loop.run_forever()

    
    async def handleRequest(self, request: web.Request):

        # Fetch body
        body = await request.json()

        # Update values
        self.pos_x = body["posX"]
        self.pos_y = body["posY"]
        self.pos_z = body["posZ"]
        
        self.speed = body["speed"]
        self.rpm = body["rpm"]
        self.gear = body["gear"]

        self.racing = body["racing"]

        # Notify listeners
        for func in self.listeners:
            func()

if __name__ == "__main__":
    HTTPServer()