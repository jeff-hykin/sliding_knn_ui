import math

import pandas
import numpy
import numpy as np
from specific_tools import Transformers, LazyDict, sliding_window

predict_path = "/Users/jeffhykin/repos/primient/data/ethylex_dry_thin.predict.csv"
data_path = "/Users/jeffhykin/repos/primient/data/ethylex_dry_thin.groups.csv"


# 
# setup
# 
kwargs = dict(
    datetime_column="date",
    max_hours_gap=4,
    window_size=10,
    importance_decay=0.7,
    output_groups=["dt1_bin_product"],
    input_importance={
        'dt1_acid_flow_gpm': 1.0,
        'surge_moisture': 1.0,
        'dt1_soda_ash_flow_hr': 1.0,
        'dt1_soda_ash_flow_scaled_hr': 1.0,
    },
    number_of_neighbors=3,
)
df = pandas.read_csv(data_path, sep=",")
test_data = pandas.read_csv(predict_path, sep=",")
# 
# cleaning
# 
def enforce_numeric_and_interpolate(df, datetime_column, output_groups, input_importance, **other):
    if datetime_column not in df.columns:
        raise Exception(f'''It appears the given datetime column {repr(datetime_column)} was not one of the available columns: {df.columns}''')
    assert len(df) != 0, "It appears the provided data has no rows"
    # get the datetime
    df[datetime_column] = pandas.to_datetime(df[datetime_column], errors='coerce')
    datetime_column_values = df[datetime_column].values
    df.dropna(subset=[datetime_column])
    assert len(df) != 0, "It appears after removing invalid datetimes, the sheet has no rows"
    df = df.set_index(datetime_column)
    df[datetime_column] = datetime_column_values
    df = Transformers.simplify_column_names(df)
    for each_column in df.columns:
        if each_column in [datetime_column, *output_groups]:
            continue
        df = Transformers.attempt_force_numeric(df, each_column)
        df[each_column] = df[each_column].interpolate(method='linear')
    
    df["__timestamp_hours"] = (df.index.values.astype('float64')/1000000000)/(60*60)
    if "__ignore" in output_groups:
        df["__ignore"] = [0]*len(df)
    for each in output_groups:
        if numpy.isnan(df[each].values).all():
            raise Exception(f"It appears that all of the {each} values are NaN")
    
    df_no_nan = df.dropna(subset=input_importance.keys())
    if len(df_no_nan) == 0:
        guide = ""
        for progress, each in input_importance.keys():
            percent = (np.isnan(df[each].values).sum()/len(df))*100
            if percent != 0:
                guide += f"\n{each}: {round(percent)}% NaN values"
        
        raise Exception(f"I removed rows with non-numeric entries, but after doing so the sheet appears to be empty:{guide}")
    return df_no_nan

df = enforce_numeric_and_interpolate(df, **kwargs)

# 
# handle gaps
# 
def add_run_index(df, datetime_column, max_hours_gap, **other):
    return Transformers.label_by_value_gaps(df, column="__timestamp_hours", max_gap_size=max_hours_gap, new_column_name="__run_index")

df = add_run_index(df, **kwargs)

def noramlize(df, value_ranges):
    for each in value_ranges.keys():
        df = Transformers.normalize(df, column=each, min_value=value_ranges[each]["min"], max_value=value_ranges[each]["max"])
    return df
    
# 
# train
# 
def train(df, input_importance, datetime_column, **other):
    value_ranges = {}
    input_columns = list(input_importance.keys())
    # scale for importance
    for each in input_columns:
        value_ranges.setdefault(each,{})
        value_ranges[each]["min"] = value_ranges.get(each,{}).get("min", df[each].values.min())
        value_ranges[each]["max"] = value_ranges.get(each,{}).get("max", df[each].values.max())
    
    df = noramlize(df, value_ranges)
    df.sort_values(by=[datetime_column], inplace=True)
    df.sort_index()
    return df, value_ranges

noramlized_df, value_ranges = train(df, **kwargs)


def predict(trained_df, incoming_df, value_ranges, input_importance, importance_decay, **other):
    incoming_df = enforce_numeric_and_interpolate(incoming_df, input_importance=input_importance, **other)
    incoming_df = noramlize(incoming_df, value_ranges)
    trained_df = pandas.DataFrame(trained_df)
    incoming_df.sort_index()
    trained_df.sort_index()
    # oldest date is index 0
    
    # apply column-level importance
    for column, importance in input_importance.items():
        trained_df[column] *= importance
        incoming_df[column] *= importance
    
    trained_df_slim  = trained_df.drop(columns=[each for each in trained_df.columns if each not in input_importance])
    incoming_df_slim = incoming_df.drop(columns=[each for each in incoming_df.columns if each not in input_importance])
    
    distance_dfs = []
    for index, each_row in enumerate(incoming_df_slim.iloc):
        distance_dfs.append(
            numpy.abs(trained_df_slim.to_numpy() - each_row.to_numpy()).sum(axis=1)
        )
    distance_dfs = numpy.vstack(distance_dfs)
    try:
        # create row weights
        history_weights = []
        window_size = len(incoming_df)
        for index in range(window_size):
            history_weights.append(
                importance_decay**index
            )
        history_weights = numpy.array(tuple(reversed(history_weights)))
        offsets = numpy.array(range(window_size))
        
        true_distances = numpy.array([math.inf]*len(trained_df))
        groups = trained_df.groupby("__run_index")
        runs = list(groups.indices.keys())
        total_index = -1
        for run_index in runs:
            group = groups.get_group(run_index)
            for _ in sliding_window(range(len(group)), window_size=window_size+1):
                total_index += 1
                things = distance_dfs[offsets,total_index+offsets] * history_weights
                # first element is oldest-timestamp element 
                true_distances[total_index] = things.sum()
                # summation = 0
                # for offset, distances_for_corrisponding_input_row, relative_importance in zip(range(window_size), distance_dfs, history_weights):
                #     local_index = total_index + offset
                #     distance = distances_for_corrisponding_input_row[local_index]
                #     summation += distance * relative_importance
                # true_distances[total_index] = summation
        trained_df["__true_distances"] = true_distances
        trained_df.sort_values(by=["__true_distances"])
        # find the 3 cloesest runs
        dict(groups.__true_distances.min())
    except Exception as error:
        import code; code.interact(local={**globals(),**locals()})
    
    import code; code.interact(local={**globals(),**locals()})
    # for each row in point df
    #     df_d = (point df - row).sum by rows
    
    # for each in sliding window(indices):
    #     multiply rows by weights
    #     multiply columns by weights
    #     sum
        
    # save as new temp column
    # sort by temp column
    # select top k
    # average between them 
    # get original indices
    # for each in top
    #     get distance
    # average between the top
    # get the next index

predict(trained_df=noramlized_df, incoming_df=test_data, value_ranges=value_ranges, **kwargs)