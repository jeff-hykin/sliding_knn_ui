# encoding: utf-8
from aiohttp import web
import socketio 
import sys
import argparse 
import math
import asyncio
from time import time as now


# 
# args setup
# 
parser = argparse.ArgumentParser(description="aiohttp server") 
parser.add_argument('--port')
args = parser.parse_args()

# globals 
debugging = False

# 
# server setup
# 
app = web.Application(client_max_size=(1024 ** 2 * 100))
routes = web.RouteTableDef()
options = {} if not debugging else dict(logger=True, engineio_logger=True)
sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins="*", **options); sio.attach(app)

if debugging: print('starting server') 

@routes.get('/')
async def index(request : web.Request): 
    if debugging: print(f'''web request''')
    try:
        with open("./main/index.html",'r') as f:
            output = f.read()
    except:
        output = None
    return web.Response(text=output, content_type='text/html')

@routes.post('/ping')
async def ping(request : web.Request):
    return web.Response(text='"pong"')

@routes.post('/set_training_data')
async def set_training_data(request : web.Request):
    try:
        post_result = await request.post()
        large_file = post_result.get("file")
        handle_incoming_training_file(large_file.file)
    except Exception as error:
        return web.Response(text=f'''{{"success":false, "error": {json.dumps(f"{error}")} }}''')
        
    return web.Response(text='''{"success":true}''')

@routes.post('/set_predict_data')
async def set_predict_data(request : web.Request):
    try:
        data = await request.text()
    except Exception as error:
        print('error = ', error)
    request
    return web.Response(text="null")

@routes.post('/run_prediction')
async def run_prediction(request : web.Request):
    try:
        data = await request.text()
    except Exception as error:
        print('error = ', error)
    request
    return web.Response(text="null")

# @routes.post('/large/set/{content_type}/{data_id}')
# async def set_large_data(request : web.Request):
#     global large_data
#     content_type = request.match_info["content_type"]
#     content_type = content_type.replace(r"%2F", "/")
#     large_data_id = request.match_info["data_id"]
#     # save in ram
#     post_result = await request.post()
#     large_file = post_result.get("file")
#     if large_file is not None:
#         large_data[large_data_id] = large_file.file.read()
#     return web.Response(text="null")

# @routes.get('/large/get/{content_type}/{data_id}')
# async def get_large_data(request : web.Request):
#     global large_data
#     content_type = request.match_info["content_type"]
#     content_type = content_type.replace(r"%2F", "/")
#     large_data_id = request.match_info["data_id"]
#     return web.Response(
#         content_type=content_type,
#         body=large_data[large_data_id],
#     )

# 
# start server
# 
app.add_routes(routes); web.run_app(app, port=args.port)