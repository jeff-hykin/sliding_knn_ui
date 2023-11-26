from __dependencies__.blissful_basics import wrap_around_get, stringify, FS, print, indent

class Colors:
    def __init__(self, color_mapping):
        self._color_mapping = color_mapping
        for each_key, each_value in color_mapping.items():
            if isinstance(each_key, str) and len(each_key) > 0 and each_key[0] != '_':
                setattr(self, each_key, each_value)
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return wrap_around_get(key, list(self._color_mapping.values()))
        elif isinstance(key, str):
            return self._color_mapping.get(key, None)
    
    def __repr__(self):
        return stringify(self._color_mapping)
    
    def __iter__(self):
        for each in self._color_mapping.values():
            yield each
    
    def values(self):
        return self._color_mapping.values()

xd_theme = Colors({
    "blue":             '#82aaff',
    "purple":           '#c792ea',
    "dim_green":        '#80cbc4',
    "pink":             '#e57eb3',
    "yellow":           '#fec355',
    "bold_green":       '#4ec9b0',
    "brown":            '#ce9178',
    "rust":             '#c17e70',
    "orange":           '#f78c6c',
    "bananna_yellow":   '#ddd790',
    "lime":             '#c3e88d',
    "green":            '#4ec9b0',
    "soft_red":         '#f07178',
    "dark_slate":       '#3f848d',
    "vibrant_green":    '#04d895',
    "black":            '#000000',
    "white":            '#ffffff',
    "light_gray":       '#c7cbcd',
    "cement":           '#698098',
    "gray":             '#546e7a',
    "light_slate":      '#64bac5',
    "light_blue":       '#89ddff',
    "electric_blue":    '#00aeff',
    "red":              '#ff5572',
})
default_theme = xd_theme

def points_to_function(x_values, y_values, are_sorted=False):
    number_of_values = len(x_values)
    if number_of_values != len(y_values):
        raise ValueError("x_values and y_values must have the same length")
    if number_of_values == 0:
        raise ValueError("called points_to_function() but provided an empty list of points")
    # horizontal line
    if number_of_values == 1:
        return lambda x_value: y_values[0]
    
    if not are_sorted:
        # sort to make sure x values are least to greatest
        x_values, y_values = zip(
            *sorted(
                zip(x_values, y_values),
                key=lambda each: each[0],
            )
        )
    
    minimum_x = x_values[0]
    maximum_x = x_values[-2] # not the true max, but, because of indexing, the 2nd-maximum
    def inner_function(x):
        if x >= maximum_x:
            # needs -2 because below will do x_values[x_index+1]
            x_index = number_of_values-2
        elif x <= minimum_x:
            x_index = 0
        else:
            # binary search for x
            low = 0
            high = number_of_values - 1

            while low < high:
                mid = (low + high) // 2

                if x_values[mid] < x:
                    low = mid + 1
                else:
                    high = mid

            if low > 0 and x < x_values[low - 1]:
                low -= 1
            
            x_index = low
        
        # Perform linear interpolation / extrapolation
        x0, x1 = x_values[x_index], x_values[x_index+1]
        y0, y1 = y_values[x_index], y_values[x_index+1]
        # verticle line
        if (x1 - x0) == 0:
            return y1
        slope = (y1 - y0) / (x1 - x0)
        y = y0 + slope * (x - x0)

        return y
    
    return inner_function

def graph_lines(*args, title, x_axis_name, y_axis_name, save_to=None, x_axis_scale='linear', y_axis_scale='linear', x_range=None, y_range=None, display=True, with_dots=True, scroll_zoom=True, drag_mode="pan"):
    """
        Example:
            graph_lines(
                dict(
                    name="line 1",
                    x_values=[0,1,2,3],
                    y_values=[0,1,1,2],
                    color="", # optional
                ),
                dict(
                    name="line 2"
                    x_values=[0,1,2,3],
                    y_values=[0,1,1,2],
                ),
                title="Linear vs. Non-Linear Energy Method",
                x_axis_name="X",
                y_axis_name="Displacement",
            )
    """
    # print(title)
    # with print.indent:
    #     print(stringify(args))
    
    if len(args) == 0:
        raise Exception(f'''\n\ngraph_lines(\n    title={title},\n    x_axis_name={x_axis_name},\n    y_axis_name={y_axis_name}\n)\nwas called without any normal args (e.g. no lines/line-data given)''')
    
    try:
        import pandas as pd
        import plotly.express as px
        x_values = []
        y_values = []
        names = []
        for line_index, each in enumerate(args):
            x_values += list(each["x_values"])
            y_values += list(each["y_values"])
            names += [each.get("name",f"line_{line_index+1}")]*len(each["x_values"])
        assert len(set([each["name"] for each in args])) == len(args), f"""When graphing multiple lines, they need to have unique names:\n{([each["name"] for each in args])}"""
        color_based_feature = "Group"
        if x_axis_name == color_based_feature or y_axis_name == color_based_feature:
            color_based_feature = "[Group]"
            if x_axis_name == color_based_feature or y_axis_name == color_based_feature:
                color_based_feature = "[[Group]]"
        null_label = " "
        if x_axis_name == null_label or y_axis_name == null_label:
            null_label = "  "
            if x_axis_name == null_label or y_axis_name == null_label:
                null_label = "   "
        data = {
            x_axis_name: x_values,
            y_axis_name: y_values,
            color_based_feature: names,
            null_label: [""]*len(names),
        }
        df = pd.DataFrame(data)
        kwargs = dict(
            x=x_axis_name,
            y=y_axis_name,
            title=title,
        )
        if len(set(names)) > 1:
            kwargs["color"] = color_based_feature
        if with_dots:
            kwargs["text"] = null_label
        fig = px.line(df, **kwargs)
        fig.update_traces(textposition="bottom right")
        fig.update_layout(xaxis_type=x_axis_scale, yaxis_type=y_axis_scale)
        for line_index, line_info in enumerate(args):
            if line_info.get("color", None):
                fig.data[line_index].line.color = line_info["color"]
        if x_range:
            fig.update_layout(xaxis_range=x_range)
        if y_range:
            fig.update_layout(yaxis_range=y_range)
        
        fig.update_layout({
            'dragmode': drag_mode,
        })
        config = {
            "scrollZoom": scroll_zoom,
        }
        if save_to:
            FS.ensure_is_folder(FS.parent_path(save_to))
            fig.write_html(save_to, config=config)
        
        if display:
            fig.show(config=config)
        
        return fig
    except Exception as error:
        raise Exception(f'''
            error in graph_lines: {error}
            function call:
                graph_lines(
                    {indent(stringify(args), by="                    ", ignore_first=True)},
                    title={repr(title)},
                    x_axis_name={repr(x_axis_name)},
                    y_axis_name={repr(y_axis_name)},
                    save_to={repr(save_to)},
                )
        ''')
    

from copy import deepcopy
def graph_groups(
    groups,
    remove_space_below_individual=False,
    group_averaging_function=None,
    theme=default_theme,
    **kwargs,
):
    groups = deepcopy(groups)
    lines = []
    for each in groups.values():
        lines += each["lines"]
    
    # 
    # group average
    # 
    if callable(group_averaging_function):
        new_lines = []
        for group_index, (group_name, each_group) in enumerate(groups.items()):
            lines = each_group["lines"]
            
            functions = [
                points_to_function(
                    each["x_values"],
                    each["y_values"],
                )
                    for each in lines
            ]
            
            # x values might not be the same across lines, so get all of them
            any_x_value = set()
            for each_line in lines:
                any_x_value |= set(each_line["x_values"])
            any_x_value = sorted(any_x_value)
            
            y_values = [
                group_averaging_function([ each_function(each_x) for each_function in functions ])
                    for each_x in any_x_value
            ]
            new_lines.append(
                dict(
                    x_values=any_x_value,
                    y_values=y_values,
                    name=group_name,
                    color=each_group.get("color", theme[group_index]),
                )
            )
        
        lines = new_lines
    # 
    # flatten
    # 
    if remove_space_below_individual:
        # find the min y value for each x
        from collections import defaultdict
        per_x_value = defaultdict(lambda:[])
        for each_line in lines:
            for each_x, each_y in zip(each_line["x_values"], each_line["y_values"]):
                per_x_value[each_x].append(each_y)
        min_per_x = {}
        for each_x, values in per_x_value.items():
            min_per_x[each_x] = min(values)
        # flatten all the data
        for each_line in lines:
            for index, (each_x, each_y) in enumerate(zip(each_line["x_values"], each_line["y_values"])):
                each_line["y_values"][index] = each_y - min_per_x[each_x]
    
    graph_lines(
        *lines,
        **kwargs,
    )


def create_slider_from_traces(traces):
    import plotly.graph_objects as go
    fig = go.Figure()
    for each in traces:
        fig.add_trace(each)

    # Create and add slider
    steps = []
    for index in range(len(fig.data)):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
                {"title": "Timestep: " + str(index)},
            ],  # layout attribute
        )
        step["args"][0]["visible"][index] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [
        dict(
            steps=steps,
        )
    ]

    fig.update_layout(sliders=sliders)
    return fig


import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

def multi_plot(figures, save_to=None, height=750):
    if save_to != None:
        # recusive call 1 time and save the output to a file
        output = multi_plot(figures, save_to=None, height=height)
        import os
        directory = os.path.dirname(save_to)
        if directory == '': directory = '.'
        os.makedirs(directory, exist_ok=True)
        with open(save_to, 'w') as file:
            file.write(
                output
            )
        return output
    
    # grid
    names = [""]*len(figures)
    if isinstance(figures, dict):
        names = figures.keys()
        figures = tuple(figures.values())
    
    if len(figures) > 0 and isinstance(figures[0], (list, tuple)):
        rows_of_graphs = []
        for each_row in figures:
            for each_figure in each_row:
                try:
                    each_figure.update_layout(height=height)
                except Exception as error:
                    import code; code.interact(local={**globals(),**locals()})
            
            rows_of_graphs.append(
                "\n".join([
                    plot(each_figure, output_type='div', include_plotlyjs=(index==0), config={'displayModeBar': False})
                        for index, each_figure in enumerate(each_row)
                ])
            )
        
        graphs = "\n".join(tuple(
            f"""
                <h3>{name}</h3>
                <div class="row">
                    {each}
                </div>
            """
                for name, each in zip(names, rows_of_graphs)
        ))
        
        return f"""
            <!DOCTYPE html>
            <html>
                <head>
                    <meta charset="utf-8">
                    <title>Two Full-Sized Plotly Graphs with Scroll Bar</title>
                    <style>
                        body {{
                            font-family: sans-serif;
                        }}
                        .row {{
                            display: flex;
                            flex-direction: row;
                            max-width: 100vw;
                            overflow-x: auto;
                            overflow-y: hidden;
                            min-height: {height}px;
                        }}
                    </style>
                </head>
                <body style="overflow: auto; display: flex; flex-direction: column; height: 100vh; min-height: 100vh; max-height: 100vh;">
                    {graphs}
                </body>
            </html>
        """
    # list
    else:    
        for each_figure in figures:
            each_figure.update_layout(height=height)
            
        graphs = "\n".join([
            plot(each_figure, output_type='div', include_plotlyjs=(index==0), config={'displayModeBar': False})
                for index, each_figure in enumerate(figures)
        ])
        return f"""
            <!DOCTYPE html>
            <html>
                <head>
                    <meta charset="utf-8">
                    <title>Two Full-Sized Plotly Graphs with Scroll Bar</title>
                </head>
                <body style="overflow: auto; display: flex; flex-direction: column; height: 100vh; min-height: 100vh; max-height: 100vh;">
                    {graphs}
                </body>
            </html>
        """


def linear_correlation_plot(x_values, y_values, title="", display=True, save_to=None):
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np
    from sklearn.linear_model import LinearRegression
        
    # Assuming x and y are your data arrays
    new_x_values = []
    new_y_values = []
    for x,y in zip(x_values, y_values):
        if x != x or y!=y:
            continue
        else:
            new_x_values.append(x)
            new_y_values.append(y)
    x = np.array(new_x_values)
    y = np.array(new_y_values)
    
    # Create a scatter plot of your data points
    scatter = go.Scatter(x=x, y=y, mode='markers', name='Data')
    
    
    plots = [ scatter ]
    if len(x) > 3:
        regression = LinearRegression()
        regression.fit(x.reshape(-1, 1), y)
        
        r_squared = regression.score(x.reshape(-1, 1), y)
        
        title = f"{title} (r^2: {r_squared})"
        # Generate predictions for the line of best fit
        x_range = np.linspace(min(x), max(x), 100)
        y_pred = regression.predict(x_range.reshape(-1, 1))

        # Create a line plot for the line of best fit
        line = go.Scatter(x=x_range, y=y_pred, mode='lines', name='Line of Best Fit')
        
        plots.append(line)

    # Create the layout for the plot
    layout = go.Layout(title=title)

    # Create the figure
    fig = go.Figure(data=plots, layout=layout)

    if save_to:
        FS.ensure_is_folder(FS.parent_path(save_to))
        fig.write_html(save_to)
    
    # Show the plot
    if display:
        fig.show()
    
    return fig



# import plotly.graph_objects as go
# import numpy as np

# # Sample data
# x = np.array([1, 2, 3, 4, 5])
# y = np.array([2, 3, 4, 4, 5])
# colors = ['red', 'green', 'blue', 'purple', 'orange']  # Define colors for each point

# # Create a scatter plot with custom colors
# scatter = go.Scatter(
#     x=x,
#     y=y,
#     mode='markers',
#     name='Data',
#     marker=dict(
#         size=10,  # Adjust the size of the markers as needed
#         color=colors,  # Specify the color for each point
#         opacity=0.7,  # Adjust opacity if needed
#         showscale=False  # Turn off the color scale legend
#     )
# )

# # Create the layout for the plot
# layout = go.Layout(
#     title='Scatter Plot with Custom Point Colors'
# )

# # Create the figure
# fig = go.Figure(data=[scatter], layout=layout)

# # Show the plot
# fig.show()
