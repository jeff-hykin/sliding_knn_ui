import math
from statistics import median, stdev
from statistics import mean as average
from collections import defaultdict
import numbers
import os

import ez_yaml
import numpy
import numpy as np
import pandas
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from generic_tools.misc import simplify_names, pascal_case_with_spaces, excel_datetime_to_unixtime_converter, get_columns, split_csv_into_column_files, interpolate, hours_to_seconds, seconds_to_hours, Duration, sliding_window
from generic_tools.plotting import graph_lines

from __dependencies__.informative_iterator import ProgressBar
from __dependencies__.blissful_basics import LazyDict, Csv, FS, super_hash, flatten, large_pickle_save, large_pickle_load, stringify, print, to_pure, flatten_once, SemiLazyMap, Map
from __dependencies__.quik_config import find_and_load
from __dependencies__ import blissful_basics
from __dependencies__ import blissful_basics as bb

numpy.set_printoptions(suppress=True) # don't use scientific notation in confusion matrices  
ez_yaml.settings["width"] = 999999

# 
# config
# 
if True:
    info = find_and_load(
        "main/info.yaml", # walks up folders until it finds a file with this name
        cd_to_filepath=True, # helpful if using relative paths
        fully_parse_args=False, # if you already have argparse, use parse_args=True instead
        show_help_for_no_args=False, # change if you want
    )
    config  = info.config
    path_to = info.absolute_path_to
    secrets = info.secrets
    stringify.onelineify_threshold = 1

# 
# 
# tools
#
#
def create_one_hot(obj):
    new_obj = {}
    one_hot_to_obj = defaultdict(str)
    num_possible_values = len(obj)
    zeros_array = np.zeros(num_possible_values, dtype=np.uint8)
    
    for index, key in enumerate(obj.keys()):
        new_obj[key] = np.copy(zeros_array)
        new_obj[key][index] = 1
        # convert to more efficient data type
        new_obj[key] = new_obj[key].astype(np.uint8)
        one_hot_to_obj[tuple(new_obj[key])] = key
    
    return (new_obj, lambda key: one_hot_to_obj[tuple(key)])


def generate_onehot(index, size):
    vec = [0]*size
    vec[index] = 1
    return vec
    
def standard_load_train_test(path=None):
    # 
    # load data
    # 
    original_df = pandas.read_csv(path, sep="\t")

    # 
    # filters
    # 
    df = original_df
    for each in config.modeling.filters:
        df = df[df[each]]
    
    y = df[info.config.feature_to_predict]
    x = df.drop(columns=[ each for each in df.columns if each not in config.modeling.selected_features ])
    assert len(x.columns) == len(config.selected_features), "Looks like one of the selected features isn't in the dataset"

    # 
    # test split 
    # 
    # 70% training and 30% test; stratify: maintain same proportion of + and - examples
    x = x.values
    y = y.values
    try:
        if config.modeling.stratify:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config.modeling.test_proportion, stratify=y)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=config.modeling.test_proportion)
    except Exception as error:
        import code; code.interact(local={**globals(),**locals()})
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    
    def test_accuracy_of(predict):
        # 
        # total
        # 
        y_pred = tuple(each for each in predict(x_test))
        y_test_positive, y_pred_positive = zip(*((y_test[index], y_pred[index]) for index, value in enumerate(y_pred) if value == info.config.positive_label_value))
        y_test_negative, y_pred_negative = zip(*((y_test[index], y_pred[index]) for index, value in enumerate(y_pred) if value == info.config.negative_label_value))
        total_accuracy    = accuracy_score(y_test, y_pred)
        positive_accuracy = accuracy_score(y_test_positive, y_pred_positive)
        negative_accuracy = accuracy_score(y_test_negative, y_pred_negative)
        
        # FIXME: make more generic
        
        return dict(total_accuracy=total_accuracy, positive_accuracy=positive_accuracy, negative_accuracy=negative_accuracy)
    
    return x_train, x_test, y_train, y_test, test_accuracy_of

def create_trainer(*, classifier, classifier_name, module_name, output_postfix=""):
    def train(x_train, x_test, y_train, y_test, test_accuracy_of):
        print(f"training {classifier_name}")
        classifier.fit(x_train, y_train)
        accuracy_info = test_accuracy_of(classifier.predict)
        return accuracy_info, classifier
    
    if module_name == '__main__':
        accuracy_info, classifier = train(*standard_load_train_test())
        pandas.DataFrame([
            accuracy_info
        ]).to_csv(f"{classifier_name}_{output_postfix}_results.tsv")
    
    return train


def bytes_to_binary(value, separator=""):
    return separator.join([f'{each:0<8b}' for each in value])
            
def nearest_neighbor_distances(base_array, neighbor_array):
    assert isinstance(base_array, numpy.ndarray)
    assert isinstance(neighbor_array, numpy.ndarray)
    
    # base_array = numpy.array([
    #     [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,],
    #     [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,],
    # ])
    # neighbor_array = numpy.array([
    #     [1,1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,],
    #     [0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,],
    # ])
    
    def packbits_64(array):
        array = numpy.packbits(array, axis=1)
        data_type_existing_size = 8
        data_type_target_size = 64
        rows, columns = array.shape
        bits_per_column = columns * data_type_existing_size
        padding_needed = math.ceil(bits_per_column / data_type_target_size)
        if padding_needed != 0:
            array = numpy.concatenate(
                (
                    array,
                    numpy.zeros((len(array), padding_needed), dtype='uint8')
                ),
                axis=1
            )
        rows, columns = array.shape
        bits_per_column = columns * data_type_existing_size
        new_shape = (rows, bits_per_column//data_type_target_size)
        return numpy.ndarray(
            new_shape,
            numpy.uint64,
            array.tobytes(),
        )
    
    packed_base     = packbits_64(base_array)
    packed_neighbor = packbits_64(neighbor_array)
    
    min_distances = []
    for _, each_row in ProgressBar(packed_base):
        distances = numpy.bitwise_xor(each_row, packed_neighbor)
        min_distance = numpy.unpackbits(numpy.packbits(distances, axis=1), axis=1).sum(axis=1).min()
        # min_distance = numpy.unpackbits(distances, axis=1).sum(axis=1).min()
        # min_distance = min(sum(int(each_cell).bit_count() for each_cell in each_row) for each_row in distances)
        min_distances.append(min_distance)
    
    return min_distances

class ColumnizedData:
    def __init__(self, folder):
        self.folder = folder
        self.name_to_path = {}
        self.df_cache = {}
        for each_path in FS.list_file_paths_in(folder):
            if each_path.endswith(".csv"):
                *folders, name, extension = FS.path_pieces(each_path)
                self.name_to_path[name] = each_path
        self.columns = tuple(self.name_to_path.keys())
    
    def __getitem__(self,key):
        if key not in self.df_cache:
            path = self.name_to_path[key]
            self.df_cache[key] = pandas.read_csv(path, sep=",")
        return self.df_cache[key]


def excel_to_csv(file_path, output_folder=None, show_warnings=True):
    # Read the Excel file
    xls = pandas.ExcelFile(file_path)
    
    # make the containing folder
    directory = output_folder or f'{os.path.dirname(file_path)}/{pascal_case_with_spaces(os.path.basename(file_path).split(".")[0]).replace(" ","_").lower()}_csv_files/'
    if directory == '': directory = '.'
    os.makedirs(directory, exist_ok=True)

    # Get the names of all sheets in the Excel file
    sheet_names = xls.sheet_names
    simplified_sheet_names = simplify_names(sheet_names, show_warnings=show_warnings)

    # Loop through each sheet and save it as a CSV file
    for progress, (sheet_name, simplified_name) in ProgressBar(tuple(zip(sheet_names, simplified_sheet_names))):
        for index in range(0,15):
            df = xls.parse(sheet_name, header=index)
            if df.columns[0] != 'Unnamed: 0':
                break
        
        names = tuple(zip(df.columns, simplify_names(df.columns)))
        df.rename(columns={
            old_name: new_name
                for old_name, new_name in names
        })

        # Save the DataFrame as a CSV file
        df.to_csv(f'{directory}/{simplified_name}.csv', index=False)




DataSources = SemiLazyMap(
)

class Transformers:
    @staticmethod
    def minutes_to_seconds(minutes):
         return minutes*60
    
    @staticmethod
    def remove_nan(df, **kwargs):
        return df.dropna(**kwargs) or df
    
    @staticmethod
    def filter_max_value(df, column, max_value=float("inf")):
        if max_value != float("inf"):
            df = df[df[column] <= max_value]
        return df
        
    @staticmethod
    def add_datetime_column(df, input_column="unix_timestamp"):
        from datetime import datetime
        df = pandas.DataFrame(df)
        df["datetime"] = tuple(datetime.fromtimestamp(each_value) for each_value in df[input_column].values)
        return df
    
    @staticmethod
    def setup_timestamp_for_interpolation(df, new_column="timestamp", unix_timestamp_column="unix_timestamp"):
        df[new_column] = pd.to_datetime(df[unix_timestamp_column], unit='s')
        df.set_index(new_column, inplace=True)
        return df
    
    @staticmethod
    def make_time_stamps_relative(df, time_column="unix_timestamp"):
        min_value = df[time_column].values.min()
        df[time_column] = df[time_column].values - min_value
        return df
    
    @staticmethod
    def convert_timestamps_from_seconds_to_hours(df, time_column="unix_timestamp"):
        df[time_column] = df[time_column].values/(60*60)
        return df
    
    @staticmethod
    def time_to_duration(df, time_column="unix_timestamp", key="seconds"):
        df[time_column] = tuple(Duration(**{key: each}) for each in df[time_column].values())
        return df
    
    @staticmethod
    def normalize(df, column, min_value=None, max_value=None):
        min_value = df[column].values.min() if type(min_value) == type(None) else min_value
        max_value = df[column].values.max() if type(max_value) == type(None) else max_value
        if max_value == min_value:
            df[column] = [0]*len(df[column])
        else:
            df[column] = (df[column].values-min_value) / (max_value - min_value)
        return df
    
    @staticmethod
    def unnormalize(df, column, min_value=None, max_value=None):
        min_value = df[column].values.min() if type(min_value) == type(None) else min_value
        max_value = df[column].values.max() if type(max_value) == type(None) else max_value
        if max_value == min_value:
            df[column] = max_value
        else:
            df[column] = ((max_value - min_value) * df[column])+min_value
        return df
    
    @staticmethod
    def column_funcs_to_lines(column_funcs):
        lines = []
        for name, each in sorted(tuple(column_funcs.items())):
            lines.append(dict(
                **getattr(each, "line_info", {}),
                x_values=(each.x_values if not hasattr(each.x_values, "values") else each.x_values.values),
                y_values=(each.y_values if not hasattr(each.y_values, "values") else each.y_values.values),
                name=name,
            ))
        return lines
    
    @staticmethod
    def simplify_column_names(df):
        from generic_tools.misc import  simplify_names, pascal_case_with_spaces
        
        old_names = df.columns
        new_names_and_values = [
            (
                pascal_case_with_spaces(each.strip()).replace(" ", "_").lower(),
                each,
                df[each].values,
            )
                for each in old_names
        ]
        names_to_drop = []
        for new_name, old_name, values in new_names_and_values:
            if new_name != old_name:
                df[new_name] = values
                names_to_drop.append(old_name)
        
        df.drop(columns=names_to_drop, inplace=True)
        return df
    
    @staticmethod
    def convert_x_to_common_hours(lines_list):
        time_min = min(
            (min(each["x_values"]) if not hasattr(each["x_values"], "min") else each["x_values"].min())
                for each in lines_list
        )
        column_name = ""
        new_lines = []
        for each_line in lines_list:
            if isinstance(each_line["x_values"], numpy.ndarray):
                x_values = (each_line["x_values"]-time_min)/(60*60)
            else:
                x_values = tuple((each-time_min)/(60*60) for each in each_line["x_values"])
                
            new_lines.append({
                **each_line,
                "x_values": x_values,
            })
            
        return new_lines
    
    @staticmethod
    def label_by_value_gaps(df, column, max_gap_size, new_column_name):
        values = df[column].values
        diffs = values[0:-1] - values[1:]
        group_index = 0
        group_values = [0]*len(values)
        for index, each_gap in enumerate(diffs):
            if each_gap == each_gap:
                if each_gap > max_gap_size:
                    group_index += 1
            group_values[index+1] = group_index
        
        df[new_column_name] = group_values
        return df
    
    @staticmethod
    def apply_value_mapping(df, column, value_mapping):
        if value_mapping:
            values = tuple(df[column].values)
            if any((each in values) for each in value_mapping.keys()):
                df[column] = tuple(value_mapping.get(each, each) for each in values)
        return df
    
    @staticmethod
    def attempt_force_numeric(df, column_name, numeric_check_size=10):
        values = df[column_name].values
        tests = []
        for each in values:
            if len(tests) > numeric_check_size:
                break
            if each != each:
                continue
            if each == 0:
                continue
            tests.append(False)
            try:
                tests[-1] = float(each) == float(each)
            except Exception as error:
                pass
        
        data_should_be_numeric = all(tests)
        if data_should_be_numeric:
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        
        return df
    
    @staticmethod
    def place_none_in_gaps(lines, max_gap_size, inplace=True):
        new_lines = []
        for each_line in lines:
            values = tuple(zip(each_line["x_values"], each_line["y_values"]))
            new_values = []
            # get pairwise elements
            for (prev_x, prev_y), (each_x, each_y) in zip(values[0:-1], values[1:]):
                new_values.append((prev_x, prev_y))
                if abs(prev_x - each_x) > max_gap_size:
                    new_values.append(((each_x+prev_x)/2, None))
            
            x_values, y_values = zip(*new_values)
            if inplace:
                each_line["x_values"] = x_values
                each_line["y_values"] = y_values
                new_lines.append(each_line)
            else:
                new_lines.append({
                    **each_line,
                    "x_values":x_values,
                    "y_values":y_values,
                })
        return new_lines
    
    @staticmethod
    def outlier_smoothing(lines, inplace=True):
        from generic_tools.misc import outlier_smoothing as smoothing
        new_lines = []
        for each_line in lines:
            if inplace:
                each_line["y_values"] = smoothing(each_line["y_values"])
                new_lines.append(each_line)
            else:
                new_lines.append({
                    **each_line,
                    "y_values":smoothing(each_line["y_values"]),
                })
        return new_lines
    
    @staticmethod
    def timeline_columns_cleaning(columnized_data, limits, x_column_name):
        def transform(each_column):
            df = columnized_data[each_column]
            column_limits = limits.get(each_column, {})
            df = Transformers.attempt_force_numeric(df, each_column)
            df = Transformers.apply_value_mapping(df, column=each_column, value_mapping=column_limits.get("value_mapping", {}))
            df = Transformers.attempt_force_numeric(df, each_column)
            df = Transformers.remove_nan(df, subset=[each_column,x_column_name], inplace=True)
            df = Transformers.filter_max_value(df, column=each_column, max_value=column_limits.get("max_value", float("inf")))
            df = Transformers.add_datetime_column(df, input_column="unix_timestamp")
            
            return df
        
        return SemiLazyMap({
            key: transform
                for key in columnized_data.columns
        })
    
    @staticmethod
    def columns_to_functions(cleaned_data, limits, x_column_name):
        def transform(each_column):
            df = cleaned_data[each_column]
            column_limits = limits.get(each_column, {})
            
            try:
                x_values = df[x_column_name].values
            except Exception as error:
                import code; code.interact(local={**globals(),**locals()})
            y_values = df[each_column].values
            inner_function = interpolate(
                x_values=x_values,
                y_values=y_values,
                max_gap=column_limits.get("max_gap", None),
                close_enough=column_limits.get("close_enough", None),
            )
            # time converter
            def function(value):
                if hasattr(value,"seconds"):
                    value = value.seconds
                return inner_function(value)
            function.df = df
            function.pure_df = pandas.DataFrame(dict(x_values=x_values, y_values=y_values))
            function.x_values = x_values
            function.y_values = y_values
            function.name = each_column
            
            return function
        
        return SemiLazyMap({
            key: transform
                for key in cleaned_data[Map.Keys]
        })
    
    @staticmethod
    def column_summary_to_limits_dict(column_summary, value_mapping):
        import numbers
        limits = {}
        for each_key, each_value in column_summary.items():
            max_gap = each_value.get("max_gap_minutes", None)
            if isinstance(max_gap, numbers.Number):
                max_gap = Transformers.minutes_to_seconds(max_gap)
            
            close_enough = each_value.get("close_enough_minutes", None)
            if isinstance(close_enough, numbers.Number):
                close_enough = Transformers.minutes_to_seconds(close_enough)
            
            limits[each_key] = dict(
                close_enough=close_enough,
                max_gap=max_gap,
                max_value=each_value.get("max_real_value", float("inf")),
                value_mapping={**each_value.get("value_mapping", {}), **value_mapping},
            )
        
        return limits
    
    @staticmethod
    def timeline_hours_plot(column_funcs, gap_size=1, x_range=None, y_range=None, title="", x_axis_name="hours", y_axis_name="", save_to=None, x_axis_scale='linear', y_axis_scale='linear', display=True, with_dots=True):
        lines = Transformers.column_funcs_to_lines(column_funcs)
        lines = Transformers.convert_x_to_common_hours(lines)
        lines = Transformers.place_none_in_gaps(lines, gap_size=gap_size)  # gap_size is in hours
        
        return graph_lines(
            *lines,
            x_range=x_range,
            y_range=y_range,
            title=title,
            x_axis_name=x_axis_name,
            y_axis_name=y_axis_name,
            save_to=save_to,
            x_axis_scale=x_axis_scale,
            y_axis_scale=y_axis_scale,
            display=display,
            with_dots=with_dots,
        )    
    
class Pipelines:
    pass