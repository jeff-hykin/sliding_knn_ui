import sys
import os
import pandas

from specific_tools import Transformers, LazyDict

global_state = LazyDict(
    
)

def handle_incoming_training_file(data_bytes):
    df = pandas.read_csv(data_bytes, sep=",")
    df = Transformers.simplify_column_names(df)
    for each_column in df.columns:
        df = Transformers.attempt_force_numeric(df, each_column)
        df[each_column] = df[each_column].interpolate(method='linear')
    global_state.training_df = df
    return df

def handle_incoming_predict_df(data_bytes):
    df = pandas.read_csv(data_bytes, sep=",")
    df = Transformers.simplify_column_names(df)
    for each_column in df.columns:
        df = Transformers.attempt_force_numeric(df, each_column)
        df[each_column] = df[each_column].interpolate(method='linear')
    global_state.predict_df = df
    return df

def run_training(conditions):
    conditions["number_of_neighbors"] = conditions.get("number_of_neighbors", 3)
    conditions, data = generate_data(
        global_state.training_df,
        **conditions,
    )
    models = {}
    for each_product, (inputs, outputs) in data.items():
        model = KNeighborsRegressor(n_neighbors=conditions["number_of_neighbors"])
        model.fit(inputs, outputs)
        models[each_product] = model
    
    global_state.conditions = conditions
    global_state.models = models
    
def run_prediction(): 
    recent_data = global_state.predict_df
    conditions = global_state.conditions
    models = global_state.models
    
    output_sampler = dict(recent_data.iloc[0])
    output_kind = tuple(output_sampler[each] for each in conditions["output_groups"]) 
    model = models[output_kind]
    # add one extra row that will be skipped
    recent_data.loc[len(recent_data)] = recent_data.iloc[-1]
    recent_data["__run_index"] = [0]*len(recent_data)
    conditions, processed_io = generate_data(recent_data, **conditions)
    inputs, outputs = processed_io[output_kind]
    prediction = model.predict(inputs[:])
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
        "dt1_bin_product",
    ],
    input_importance={
        'dt1_acid_flow_gpm': 1.0,
        'surge_moisture': 1.0,
        'dt1_soda_ash_flow_hr': 1.0,
        'dt1_soda_ash_flow_scaled_hr': 1.0,
    },
    value_ranges={},
    **kwargs,
):
    conditions = dict(
        window_size=window_size,
        importance_decay=importance_decay,
        output_groups=output_groups,
        input_importance=input_importance,
        value_ranges=value_ranges,
        **kwargs,
    )
    # get the datetime
    df[datetime_column] = pandas.to_datetime(df[datetime_column], errors='coerce')
    df.set_index(datetime_column, inplace=True)
    
    df = Transformers.simplify_column_names(df)
    for each_column in df.columns:
        df = Transformers.attempt_force_numeric(df, each_column)
        df[each_column] = df[each_column].interpolate(method='linear')

    df["__timestamp_hours"] = (df[column].values.astype('float64')/1000000000)/(60*60)
    df = Transformers.label_by_value_gaps(df, column="__timestamp_hours", max_gap_size=max_hours_gap, new_column_name="__run_index")
    
    input_columns = list(input_importance.keys())
    # scale for importance
    for each in input_columns:
        value_ranges.setdefault(each,{})
        value_ranges[each]["min"] = value_ranges.get(each,{}).get("min", df[each].values.min())
        value_ranges[each]["max"] = value_ranges.get(each,{}).get("max", df[each].values.max())
        df = Transformers.normalize(df, column=each, min_value=value_ranges[each]["min"], max_value=value_ranges[each]["max"])
        df[each] = df[each]*input_importance[each]
    
    io_for_product = {}
    grouped = df.groupby(by=[*output_groups,'__run_index'])
    for (*output_groups, run_index) in grouped.indices.keys():
        if output_group not in io_for_product:
            inputs = {
                f"{column}_{index}": []
                    for index in range(window_size)
                        for column in input_columns
            }
            outputs = {
                column: []
                    for column in input_columns
            }
            io_for_product[output_group] = (inputs, outputs)
        inputs, outputs = io_for_product[output_group]
        group = grouped.get_group(((dt1_bin_product, run_index)))
        for window in sliding_window(group.iloc, window_size=window_size+1):
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
    return conditions, {
        key: (pandas.DataFrame(inputs), pandas.DataFrame(outputs))
            for key, (inputs, outputs) in io_for_product.items()
    }

