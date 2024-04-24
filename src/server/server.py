from aiohttp import web

pos_x = 0
pos_y = 0
pos_z = 0

speed = 0
rpm = 0
gear = 0

racing = False

async def handleRequest(request: web.Request):
    body = await request.json()
    pos_x = body["posX"]
    pos_y = body["posY"]
    pos_z = body["posZ"]
    
    speed = body["speed"]
    rpm = body["rpm"]
    gear = body["gear"]

    racing = body["racing"]
    
    print(body)

app = web.Application()
app.add_routes([web.post('/', handleRequest)])


def startHTTPServer():
    web.run_app(app, port=6969)

if __name__ == "__main__":
    startHTTPServer()