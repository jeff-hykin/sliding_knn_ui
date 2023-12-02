import sys
import os

import numpy
import numpy as np
import numbers
import pandas
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from cool_cache import cache, super_hash

from specific_tools import Transformers, LazyDict, sliding_window
from informative_iterator import ProgressBar

global_state = LazyDict(
    
)

def preprocess_dataframe(data_bytes):
    df = pandas.read_csv(data_bytes, sep=",")
    df = Transformers.simplify_column_names(df)
    for each_column in df.columns:
        df = Transformers.attempt_force_numeric(df, each_column)
        df[each_column] = df[each_column].interpolate(method='linear')
    
    if len(df) == 0:
        raise Exception("The uploaded sheet appears to be empty")
    return df
    

def handle_incoming_training_file(data_bytes):
    df = preprocess_dataframe(data_bytes)
    global_state.training_df = df
    global_state.training_df_hash = super_hash(df.to_csv())
    return df

def handle_incoming_predict_df(data_bytes):
    df = preprocess_dataframe(data_bytes)
    global_state.predict_df = df
    return df

# @cache(depends_on=lambda *args,**kwargs: (global_state.training_df_hash, global_state.conditions_hash,))
def run_training():
    conditions = global_state.conditions
    conditions["number_of_neighbors"] = conditions.get("number_of_neighbors", 3)
    conditions, data = generate_data(
        global_state.training_df,
        **conditions,
    )
    models = {}
    print("training model")
    print(f"data.keys():{data.keys()}")
    try:
        for each_product, (inputs, outputs) in data.items():
            model = KNeighborsRegressor(n_neighbors=conditions["number_of_neighbors"])
            model.fit(inputs, outputs)
            models[each_product] = model
    except:
        import code; code.interact(local={**globals(),**locals()})
    print("finished training")
    return conditions, models
    
def run_prediction():
    recent_data = global_state.predict_df
    conditions = global_state.conditions
    models = global_state.models
    print("models",models)
    
    try:
        output_sampler = dict(recent_data.iloc[0])
        output_kind = tuple(output_sampler[each] for each in conditions["output_groups"]) 
        output_kind = output_kind or (0,)
        model = models[output_kind]
        # add one extra row that will be skipped
        recent_data.loc[len(recent_data)] = recent_data.iloc[-1]
        recent_data["__run_index"] = [0]*len(recent_data)
        conditions, processed_io = generate_data(recent_data, **conditions, for_predict=True)
        inputs, outputs = processed_io[output_kind]
        print("predicting")
        prediction = model.predict(inputs[:])
    except:
        import code; code.interact(local={**globals(),**locals()})
    print("predicted")
    return {
        name: value
            for name, value in zip(outputs.columns, prediction[0])
    }

def generate_data(
    df,
    *,
    datetime_column="datetime",
    max_hours_gap=4,
    window_size=10,
    importance_decay=0.7,
    output_groups=[
    ],
    input_importance={
        'dt1_acid_flow_gpm': 1.0,
        'surge_moisture': 1.0,
        'dt1_soda_ash_flow_hr': 1.0,
        'dt1_soda_ash_flow_scaled_hr': 1.0,
    },
    value_ranges={},
    for_predict=False,
    **kwargs,
):
    # importance_decay = global_state.conditions["importance_decay"]
    conditions = dict(
        datetime_column=datetime_column,
        max_hours_gap=max_hours_gap,
        window_size=window_size,
        importance_decay=importance_decay,
        output_groups=output_groups,
        input_importance=input_importance,
        value_ranges=value_ranges,
        **kwargs,
    )
    if len(output_groups) == 0 or "__ignore" in output_groups:
        output_groups = ("__ignore",)
    # get the datetime
    df[datetime_column] = pandas.to_datetime(df[datetime_column], errors='coerce')
    df = df.set_index(datetime_column)
    df = Transformers.simplify_column_names(df)
    for each_column in df.columns:
        if each_column in [datetime_column, *output_groups]:
            continue
        df = Transformers.attempt_force_numeric(df, each_column)
        df[each_column] = df[each_column].interpolate(method='linear')
    # dd
    df["__timestamp_hours"] = (df.index.values.astype('float64')/1000000000)/(60*60)
    if "__ignore" in output_groups:
        df["__ignore"] = [0]*len(df)
    for each in output_groups:
        if numpy.isnan(df[each].values).all():
            raise Exception(f"It appears that all of the {each} values are NaN")
    
    df = Transformers.label_by_value_gaps(df, column="__timestamp_hours", max_gap_size=max_hours_gap, new_column_name="__run_index")
    
    input_columns = list(input_importance.keys())
    # scale for importance
    for each in input_columns:
        value_ranges.setdefault(each,{})
        value_ranges[each]["min"] = value_ranges.get(each,{}).get("min", df[each].values.min())
        value_ranges[each]["max"] = value_ranges.get(each,{}).get("max", df[each].values.max())
        df = Transformers.normalize(df, column=each, min_value=value_ranges[each]["min"], max_value=value_ranges[each]["max"])
        df[each] = df[each]*input_importance[each]
    
    df_no_nan = df.dropna(subset=input_importance.keys())
    import numpy as np
    if len(df_no_nan) == 0:
        guide = ""
        for progress, each in ProgressBar(input_importance.keys()):
            percent = (np.isnan(df[each].values).sum()/len(df))*100
            if percent != 0:
                guide += f"\n{each}: {round(percent)}% NaN values"
        
        raise Exception(f"I removed rows with non-numeric entries, but after doing so the sheet appears to be empty:{guide}")
    df = df_no_nan
    io_for_product = {}
    print("df",df)
    try:
        grouped = df.groupby(by=[*output_groups,'__run_index'])
        print("grouped", grouped)
        global_state.grouped = grouped
        global_state.output_groups = output_groups
        for (*output_group, run_index) in grouped.indices.keys():
            output_group = tuple(output_group)
            inputs = {
                f"{column}_{index}": []
                    for index in range(window_size-1)
                        for column in input_columns
            }
            outputs = {
                column: []
                    for column in input_columns
            }
            io_for_product[output_group] = (inputs, outputs)
            inputs, outputs = io_for_product[output_group]
            group = grouped.get_group(((*output_group, run_index)))
            window_size_bump = -1 if for_predict else 0
            for window in sliding_window(group.iloc, window_size=window_size_bump):
                last_value = window.pop()
                each_input = []
                relative_importance = 1.0
                for index, each in enumerate(reversed(window)):
                    row = dict(each)
                    for each_input in input_columns:
                        inputs[f"{each_input}_{index}"].append(
                            row[each_input] * relative_importance
                        )
                    relative_importance *= importance_decay
                
                for each_input in input_columns:
                    outputs[each_input].append(
                        last_value[each_input]
                    )
        assert len(io_for_product) != 0
    except:
        import code; code.interact(local={**globals(),**locals()})
    
    return conditions, {
        key: (pandas.DataFrame(inputs), pandas.DataFrame(outputs))
            for key, (inputs, outputs) in io_for_product.items()
    }

