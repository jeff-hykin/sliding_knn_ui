# encoding: utf-8
from aiohttp import web
import socketio 
import sys
import argparse 
import math
import asyncio
from time import time as now
import pandas
import simplejson as json

from modeling import handle_incoming_training_file, handle_incoming_predict_df, run_training, run_prediction
# TODO:
    # render output response
    # add warnings/errors

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
    output = ""
    try:
        import os
        with open("./main/index.html",'r') as f:
            output = f.read()
    except Exception as error:
        try:
            with open("./index.html",'r') as f:
                output = f.read()
        except Exception as error:
            print(f'''error = {error}''')
    return web.Response(text=output, content_type='text/html')

@routes.post('/ping')
async def ping(request : web.Request):
    return web.Response(text='"pong"')

@routes.post('/set_training_data')
async def set_training_data(request : web.Request):
    try:
        post_result = await request.post()
        large_file = post_result.get("file")
        df = handle_incoming_training_file(large_file.file)
    except Exception as error:
        return web.Response(text=f'''{{"success":false, "error": {json.dumps(f"{error}")} }}''')
        
    columns = df.columns.tolist()
    data = [ each.tolist() for each in df.iloc[0:10].values ]
    result = json.dumps(dict(columns=columns,data=data, ), ignore_nan=True)
    return web.Response(text=f'''{{"success":true, "preview":{result} }}''')

@routes.post('/set_predict_data')
async def set_predict_data(request : web.Request):
    try:
        post_result = await request.post()
        large_file = post_result.get("file")
        df = handle_incoming_predict_df(large_file.file)
    except Exception as error:
        return web.Response(text=f'''{{"success":false, "error": {json.dumps(f"{error}")} }}''')
    
    columns = df.columns.tolist()
    data = [ each.tolist() for each in df.iloc[0:10].values ]
    result = json.dumps(dict(columns=columns,data=data, ), ignore_nan=True)
    return web.Response(text=f'''{{"success":true, "preview":{result} }}''')

@routes.post('/run_prediction')
async def run_prediction_endpoint(request : web.Request):
    try:
        data = await request.text()
        values = json.loads(data)
        kwargs = {}
        kwargs['number_of_neighbors']  = values['numberOfNeighbors']
        kwargs['datetime_column']      = values['datetimeColumn']
        kwargs['max_hours_gap']        = values['maxHoursGap']
        kwargs['window_size']          = values['windowSize']
        kwargs['importance_decay']     = values['importanceDecay']
        kwargs['output_groups']        = values['outputGroups']
        kwargs['input_importance']     = values['inputImportance']
        print(f'''values = {values}''')
        run_training(kwargs)
        output = run_prediction()
    except Exception as error:
        print('error = ', type(error))
        print('error = ', error)
        raise error
        return web.Response(text=json.dumps(dict(error=f"{error}"), ignore_nan=True))
    
    return web.Response(text=json.dumps(output, ignore_nan=True))

# 
# start server
# 
app.add_routes(routes); web.run_app(app, port=args.port)