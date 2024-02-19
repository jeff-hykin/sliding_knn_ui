import math

import pandas
import numpy
import numpy as np
from copy import copy
from specific_tools import Transformers, LazyDict, sliding_window

predict_path = "/Users/jeffhykin/repos/primient/data/ethylex_dry_thin.predict.csv"
data_path = "/Users/jeffhykin/repos/primient/data/ethylex_dry_thin.groups.csv"

from blissful_basics import LazyDict
settings = LazyDict(
    default_warn=lambda each: each,
)
predictors = LazyDict().setdefault(lambda key: Predictor(namespace=key))
class Predictor:
    def __init__(self, namespace, warn=None):
        self.warn = warn or settings.default_warn
        self.namespace = namespace
        self.kwargs = {}
        self.value_ranges = {}
        self.historic_runs_df = None
        self.recent_run_df = None
        self.prediction = None
    
    def set_options(self, **kwargs):
        self.kwargs = kwargs
        return self
    
    def load_historic_data(self, file):
        if not isinstance(file, pandas.DataFrame):
            df = pandas.read_csv(file, sep=",")
        else:
            df = file
        historic_df = Predictor.enforce_numeric_and_interpolate(df, **self.kwargs)
        historic_runs_df = Predictor.add_run_index(historic_df, **self.kwargs)
        historic_runs_df, value_ranges = Predictor.train(historic_runs_df, **self.kwargs)
        self.historic_runs_df = historic_runs_df
        self.value_ranges = value_ranges
        return self
    
    def load_recent_data(self, file):
        if not isinstance(file, pandas.DataFrame):
            df = pandas.read_csv(file, sep=",")
        else:
            df = file
        recent_run_df = Predictor.enforce_numeric_and_interpolate(df, **self.kwargs)
        if isinstance(self.historic_runs_df, pandas.DataFrame):
            recent_run_df.drop(columns=[each for each in recent_run_df.columns if each not in self.historic_runs_df.columns], inplace=True)
        self.recent_run_df = recent_run_df
        return self
    
    def get_nearest(self):
        self.prediction = self.predict(
            historic_df=self.historic_runs_df,
            recent_df=self.recent_run_df,
            value_ranges=self.value_ranges,
            **self.kwargs
        )
        return self
    
    @staticmethod
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
            
            try:
                df[each_column] = pandas.to_numeric(df[each_column])
                df[each_column] = df[each_column].interpolate(method='linear')
            except Exception as error:
                count_before = len(df[each_column])
                coerced = pandas.to_numeric(df[each_column], errors="coerce")
                count_after  = len(coerced.dropna())
                propotion_of_nan_values = 1 - (count_after/count_before)
                if propotion_of_nan_values > 0.5:
                    if each_column in input_importance:
                        raise Exception(f'''Over {round(propotion_of_nan_values*100)}% of the {each_column} are non-numeric. Please remove some of the non-numeric entries''')
                    else:
                        # warn(f'''Over {round(propotion_of_nan_values*100)}% of the {each_column} are non-numeric. So I'm dropping that column''')
                        df.drop(columns=[each_column], inplace=True)
                    
        
        
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
    
    @staticmethod
    def add_run_index(df, datetime_column, max_hours_gap, **other):
        return Transformers.label_by_value_gaps(df, column="__timestamp_hours", max_gap_size=max_hours_gap, new_column_name="__run_index")
        
    @staticmethod
    def train(df, input_importance, datetime_column, **other):
        value_ranges = {}
        input_columns = list(input_importance.keys())
        # scale for importance
        for each in input_columns:
            value_ranges.setdefault(each,{})
            value_ranges[each]["min"] = value_ranges.get(each,{}).get("min", df[each].values.min())
            value_ranges[each]["max"] = value_ranges.get(each,{}).get("max", df[each].values.max())
        
        df = Predictor.noramlize(df, value_ranges)
        # df.sort_values(by=[datetime_column], inplace=True)
        df.sort_index()
        return df, value_ranges
    
    @staticmethod
    def noramlize(df, value_ranges):
        for each in (each for each in value_ranges.keys() if each not in [ "__timestamp_hours", "__run_index", "date", "unix_timestamp" ]):
            df = Transformers.normalize(df, column=each, min_value=value_ranges[each]["min"], max_value=value_ranges[each]["max"])
        return df
    
    @staticmethod
    def predict(historic_df, recent_df, value_ranges, input_importance, importance_decay, datetime_column, **other):
        recent_df_noramlized = copy(recent_df)
        historic_df_normalized = copy(historic_df)
        Predictor.noramlize(recent_df_noramlized, value_ranges)
        Predictor.noramlize(historic_df_normalized, value_ranges)
        recent_df_noramlized.sort_index()
        historic_df_normalized.sort_index()
        # oldest date is index 0
        
        # apply column-level importance
        for column, importance in input_importance.items():
            historic_df_normalized[column] *= importance
            recent_df_noramlized[column] *= importance
        
        historic_df_slim  = historic_df_normalized.drop(columns=[each for each in historic_df_normalized.columns if each not in input_importance])
        recent_df_slim = recent_df_noramlized.drop(columns=[each for each in recent_df_noramlized.columns if each not in input_importance])
        
        distance_dfs = []
        for index, each_row in enumerate(recent_df_slim.iloc):
            distance_dfs.append(
                numpy.abs(historic_df_slim.to_numpy() - each_row.to_numpy()).sum(axis=1)
            )
        distance_dfs = numpy.vstack(distance_dfs)
        try:
            # create row weights
            history_weights = []
            window_size = len(recent_df)
            for index in range(window_size):
                history_weights.append(
                    importance_decay**index
                )
            history_weights = numpy.array(tuple(reversed(history_weights)))
            offsets = numpy.array(range(window_size))
            
            true_distances = numpy.array([math.inf]*len(historic_df_normalized))
            groups = historic_df_normalized.groupby("__run_index")
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
            historic_df_normalized["__true_distances"] = true_distances
            historic_df_normalized.sort_values(by=["__true_distances"])
            
            # find the 3 cloesest runs
            minimum = min(dict(groups["__true_distances"].min()).values())
            items = dict(groups["__true_distances"].min()).items()
            
            closest_run_index = None
            for each_key, each_value in items:
                if each_value == minimum:
                    closest_run_index = each_key
                    break
            
            closest_run = historic_df_normalized[historic_df["__run_index"] == closest_run_index]
            min_index = closest_run["__true_distances"].tolist().index(closest_run["__true_distances"].min())
            next_value = []
            for index_addition in range(window_size):
                try:
                    next_value.append(
                        closest_run.iloc[min_index+1+index_addition]
                    )
                except Exception as error:
                    break
            date_times = pandas.DataFrame(next_value)[datetime_column].values
            # get the index of the historic_df where the datetime_column is equal to date_times[0]
            index = historic_df_normalized[datetime_column].tolist().index(date_times[0])
            future_slice = LazyDict(historic_df[index:index+len(date_times)].to_dict())
            # next_recommended_value = closest_run.iloc[min_index+1+index_addition]
            # import code; code.interact(local={**globals(),**locals()})
            return future_slice
        except Exception as error:
            print()
            print(f'''error = {error}''')
            print()
            import code; code.interact(local={**globals(),**locals()})
        
    pass

# 
# setup
# 
kwargs = dict(
    datetime_column="date",
    max_hours_gap=4,
    window_size=10,
    importance_decay=0.7,
    output_groups=[],
    input_importance={
        'dt1_acid_flow_gpm': 1.0,
        'surge_moisture': 1.0,
        'dt1_soda_ash_flow_hr': 1.0,
        'dt1_soda_ash_flow_scaled_hr': 1.0,
    },
    number_of_neighbors=3,
)

eth = predictors["eth"].set_options(**kwargs)
eth.set_options(**kwargs)
eth.load_historic_data(data_path)
eth.load_recent_data(predict_path)
eth.get_nearest()