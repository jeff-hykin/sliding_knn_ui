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

original_dumps = json.dumps
json.dumps = lambda *args, **kwargs: original_dumps(*args, **{"ignore_nan": True, "default":encode, **kwargs},)
dict_values = type({}.values())
dict_keys = type({}.keys())
def encode(obj):
     if isinstance(obj, pandas.Timestamp):
         return str(obj)
     if isinstance(obj, (dict_values, dict_keys, tuple, set, frozenset, range, map, filter, zip, enumerate)):
         return list(obj)
     raise TypeError(repr(obj) + " is not JSON serializable")

from specific_tools import Transformers, LazyDict
from neo import settings, predictors
# TODO:
    # use neo.py so that other columns are gained
    # NOTE: when new settings changed, bust the model cache
    # create a namespace
        # homepage picks a namespace
            # settings/namespace
            # output/namespace
                # have output.html refresh the backend every so often (setInterval)
                # have a column filter
        # create endpoint for updating settings and historical data
    # add warnings/errors

# 
# args setup
# 
parser = argparse.ArgumentParser(description="aiohttp server") 
parser.add_argument('--port')
args = parser.parse_args()

# globals 
debugging = False
runtime = LazyDict(
    training_data_df=None,
    predict_data_df=None,
)

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

settings.default_warn = warn

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

# DONE
@routes.post('/set_training_data')
async def set_training_data(request : web.Request):
    try:
        post_result = await request.post()
        large_file = post_result.get("file")
        df = pandas.read_csv(large_file.file, sep=",")
        df = Transformers.simplify_column_names(df)
        runtime.training_data_df = df
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
        df = pandas.read_csv(large_file.file, sep=",")
        df = Transformers.simplify_column_names(df)
        runtime.predict_data_df = df
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
        values = json.loads(data)
        kwargs = dict(
            datetime_column      = values['datetimeColumn'],
            number_of_neighbors  = values.get('numberOfNeighbors', 1),
            window_size          = values.get('windowSize', 5),
            max_hours_gap        = values['maxHoursGap'],
            importance_decay     = values['importanceDecay'],
            output_groups        = values['outputGroups'],
            input_importance     = values['inputImportance'],
        )
        # example values:
        #   kwargs = dict(
        #       datetime_column="date",
        #       max_hours_gap=4,
        #       window_size=10,
        #       importance_decay=0.7,
        #       output_groups=[],
        #       input_importance={
        #           'dt1_acid_flow_gpm': 1.0,
        #           'surge_moisture': 1.0,
        #           'dt1_soda_ash_flow_hr': 1.0,
        #           'dt1_soda_ash_flow_scaled_hr': 1.0,
        #       },
        #       number_of_neighbors=3,
        #   )

        eth = predictors["eth"].set_options(**kwargs)
        eth.set_options(**kwargs)
        eth.load_historic_data(runtime.training_data_df)
        eth.load_recent_data(runtime.predict_data_df)
        results = eth.get_nearest().prediction
        column_names = list(results.keys())
        column_names =  [ each for each in list(runtime.training_data_df.columns) if each in column_names ]
        rows = []
        columns = [ results[each].values() for each in column_names ]
        rows = zip(*columns)
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
    
    return web.Response(text=json.dumps(dict(success=True, data=dict(column_names=column_names, rows=rows)), ignore_nan=True))

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