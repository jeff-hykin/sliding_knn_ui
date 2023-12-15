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

from modeling import global_state, handle_incoming_training_file, handle_incoming_predict_df, run_training, run_prediction, super_hash
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
warning_message_buffer = []
def warn(message):
    warning_message_buffer.append(message)

async def send_warnings():
    warning_messages = list(warning_message_buffer)
    warning_message_buffer.clear()
    for each in warning_messages:
        await sio.emit('backend_warning', each)

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
    await send_warnings()
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
    await send_warnings()
    columns = df.columns.tolist()
    data = [ each.tolist() for each in df.iloc[0:10].values ]
    result = json.dumps(dict(columns=columns,data=data, ), ignore_nan=True)
    return web.Response(text=f'''{{"success":true, "preview":{result} }}''')

@routes.post('/run_prediction')
async def run_prediction_endpoint(request : web.Request):
    try:
        data = await request.text()
        conditions_hash = super_hash(data)
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
        global_state.conditions_hash = conditions_hash
        global_state.conditions = kwargs
        global_state.conditions, global_state.models = run_training()
        output = run_prediction()
        await send_warnings()
    except Exception as error:
        import traceback
        print('error:', error)
        print(''.join(traceback.format_exception(error)))
        if "'LazyDict' object has no attribute 'training_df_hash'" in f"{error}":
            return web.Response(text=json.dumps(dict(success=False, error=f"Need to upload training data first"), ignore_nan=True))
        elif "'LazyDict' object has no attribute 'predict_df'" in f"{error}":
            return web.Response(text=json.dumps(dict(success=False, error=f"Need to upload recent data first"), ignore_nan=True))
        else:
            return web.Response(text=json.dumps(dict(success=False, error=f"{error}"), ignore_nan=True))
    
    return web.Response(text=json.dumps(dict(success=True, data=output), ignore_nan=True))


@routes.post('/start_repl')
async def replset_predict_data(request : web.Request):
    import code; code.interact(local={**globals(),**locals()})
    return web.Response(text="{}", ignore_nan=True)

def traceback_to_string(traceback):
    import traceback as traceback_module
    from io import StringIO
    string_stream = StringIO()
    traceback_module.print_tb(traceback, limit=None, file=string_stream)
    return string_stream.getvalue()

def get_trace(level=0):
    import sys
    import types
    try:
        raise Exception(f'''''')
    except:
        traceback = sys.exc_info()[2]
        back_frame = traceback.tb_frame
        for each in range(level+1):
            back_frame = back_frame.f_back
    traceback = types.TracebackType(
        tb_next=None,
        tb_frame=back_frame,
        tb_lasti=back_frame.f_lasti,
        tb_lineno=back_frame.f_lineno
    )
    return traceback

    
# 
# start server
# 
app.add_routes(routes); web.run_app(app, port=args.port)