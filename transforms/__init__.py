import html
import numpy as np
import pandas as pd
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from IPython.core.display import display, HTML


def adapter(fn):
    def wrapper(row):
        try:
            return fn(row)
        except KeyError as e:
            warnings.warn(f"No key for {e}")
            return 0
        except ValueError as e:
            return None
        except ZeroDivisionError as e:
            return None

    return wrapper


def format(x, pos):
    m = 0
    while abs(x) >= 1000:
        m += 1
        x = x / 1000.0
    return "%.2f%s" % (x, ["", "K", "M", "B", "T"][m])



def transpose(ds):
    new_index = ds.columns[0]
    rc = ds.set_index(new_index).transpose()
    rc.reset_index(level=0, inplace=True)
    rc.rename(columns={"index": rc.columns.name}, inplace=True)
    rc.columns.name = ""
    return rc


def column_ratio_new(ds, new_col, dividend, divisor):
    return divide_new(ds, new_col, dividend, divisor)


def subtract_new(ds, new_col, minuend, subtrahend):
    rc = ds
    rc[new_col] = ds[minuend] - ds[subtrahend]
    return rc


def multiply_new(ds, new_col, multiplicand_1, multiplicand_2):
    rc = ds
    if type(multiplicand_1) == str:
        multiplicand_1 = ds[multiplicand_1]
    if type(multiplicand_2) == str:
        multiplicand_2 = ds[multiplicand_2]
    rc[new_col] = multiplicand_1 * multiplicand_2
    return rc


def add_new(ds, new_col, addend_1, addend_2):
    rc = ds
    rc[new_col] = ds[addend_1] + ds[addend_2]
    return rc


def divide_new(ds, new_col, dividend, divisor):
    rc = ds
    if dividend in ds.columns and divisor in ds.columns:
        rc[new_col] = ds[dividend] / ds[divisor]
    else:
        rc[new_col] = pd.Series(dtype="float")
    return rc


def total_column_sum_new(ds, new_col):
    rc = ds
    rc[new_col] = ds.sum(axis=1)
    return rc


def markdown_link_new(ds, new_col, link, title):
    rc = ds
    i = 0

    final_col_name = new_col
    while final_col_name in rc.columns:
        i += 1
        final_col_name = f"{new_col}:{i}"

    rc[final_col_name] = f"[{ds[title]}]({ds[link]})"
    return rc


def aggregation_new(ds, new_col, source, operation):
    if operation != "sum":
        raise Exception(f"Aggregating with {operation} is not supported")
    rc = ds
    rc[new_col] = ds[source].sum()
    return rc


def running_total_new(ds, new_col, source):
    rc = ds
    rc[new_col] = ds[source].cumsum()
    return rc


def ratio_of_total_new(ds, new_col, source):
    rc = ds
    rc[new_col] = ds[source] / ds[source].sum()
    return rc


def datediff_new(ds, new_col, start, end, increment):
    if increment == "day":
        inc = "D"
    elif increment == "week":
        inc = "W"
    elif increment == "month":
        inc = "M"
    elif increment == "year":
        inc = "Y"

    rc = ds
    rc[new_col] = rc[end] - rc[start]
    rc[new_col] = rc[new_col] / np.timedelta64(1, inc)
    rc[new_col] = rc[new_col].apply(np.floor)
    return rc


def substr_new(ds, new_col, source, start=None, end=None):
    rc = ds

    if start is None:
        start = 0
    else:
        start = start - 1

    if end is not None:
        end = start + end


    converted = rc[source].apply(lambda x: str(x) if x else '')
    rc[new_col] = converted.str[start:end]
    return rc


def custom_new(ds, new_col, function):
    rc = ds
    rc[new_col] = ds.apply(adapter(function), axis=1, result_type="reduce")
    return rc


def sqlite_new(ds, new_col, query, column_types=None):
    """
    Executes 'query' against the data as a sqlite database
    and returns the result as a new column
    """

    # Import data into an in-memory sqlite instance
    with sqlite3.connect(":memory:") as conn:
        table_name = "ds"
        ds.to_sql(name=table_name, con=conn, index=False)

        # Run new SQL query and add custom code as new column
        sql = f"SELECT *, {query} as '{new_col}' from {table_name}"
        rc = pd.read_sql(sql, conn)

    return rc


def case_statement_new(
    ds, new_col, source, conditions, default, default_type="LITERAL"
):
    rc = ds

    if default_type == "COLUMN":
        rc[new_col] = rc[default]
    else:
        rc[new_col] = default

    for condition in conditions:
        value = condition["value"]
        value_type = condition["value_type"]
        operand = condition["operand"]
        operator = condition["operator"]

        if value_type == "COLUMN":
            value = rc[value]

        # determine the operand type

        if operator == "LIKE":
            # None of the LIKE operators in the data have percent signs
            operator = "="

        # the data doesn't contain NOT LIKE

        # TODO: determine whether values should be compared using int or str

        if operator in ["=", "!=", ">", ">=", "<", "<="]:
            fn = {
                "=": "eq",
                "!=": "ne",
                ">": "gt",
                ">=": "ge",
                "<": "lt",
                "<=": "le",
            }
            f = getattr(rc[source], fn[operator])
            rc.loc[f(operand), new_col] = value

        elif operator == "IS NOT" and operand == "null":
            rc.loc[pd.notnull(rc[source]), new_col] = value

        elif operator == "IS" and operand == "null":
            rc.loc[pd.isnull(rc[source]), new_col] = value

        elif operator == "IS" and operand == "''":
            rc.loc[rc[source] == "", new_col] = value

        else:
            raise Exception(f"{operator} is not supported")

    return rc


def add(ds, col, addend):
    return add_new(ds, col, col, addend)


def multiply(ds, col, multiplicand):
    return multiply_new(ds, col, col, multiplicand)


def divide(ds, col, divisor):
    return divide_new(ds, col, col, divisor)


def round(ds, col, places):
    rc = ds
    rc[col] = ds[col].round(places)
    return rc


def substr(ds, col, start=None, end=None):
    return substr_new(ds, col, col, start, end)


def format(ds, col, arg):
    if arg != 0:
        raise Exception(f"edit_column_format does not work for non-zero args {arg}")
    rc = ds
    rc[col] = ds[col].apply(np.floor)
    return rc


def custom(ds, col, function):
    return custom_new(ds, col, adapter(function))


def sqlite(ds, col, query, column_types=None):
    # Perform the computation as a new temporary column,
    # then replace the old column with the new data
    temp_column = "hotswap_temp"

    rc = sqlite_new(ds, temp_column, query, column_types=column_types)
    rc[col] = rc[temp_column]

    return remove_columns(rc, [temp_column])


def combine_columns(
    ds, new_col, columns, separator=",", operator="concatenate", hide_columns=False
):
    rc = ds

    if operator == "concatenate":
        for col in columns:
            rc[col] = rc[col].astype(str)
        rc[new_col] = rc[columns].agg(separator.join, axis=1)

    elif operator in ("add", "subtract", "multiply"):
        for col in columns:
            rc[col] = rc[col].astype(float)

        cols = [ds[col] for col in columns]
        rc[new_col] = cols[0]
        for col in cols[1:]:
            if operator == "add":
                rc[new_col] = rc[new_col].add(col)
            elif operator == "subtract":
                rc[new_col] = rc[new_col].sub(col)
            elif operator == "multiply":
                rc[new_col] = rc[new_col].multiply(col)

    else:
        raise Exception(operator + " is unsupported")

    if hide_columns:
        rc = ds.drop(columns, errors="ignore")

    return rc


# use_three_columns is not implemented
def unpivot(ds, group_alias, values_alias):
    rc = ds.transpose().reset_index()
    rc.rename(columns={"index": group_alias, 0: values_alias}, inplace=True)
    return rc


def zero_fill(ds, column_types):
    rc = ds

    col_1_name = ds.columns[0]
    col_1_type = column_types[col_1_name]

    if col_1_type in ("text", "real", "integer"):
        # Do nothing for this zero_fill since the
        # first column is a string or real number
        pass

    elif col_1_type == "date":
        # Do something for this zero_fill since the first column is a date
        col_1 = rc[col_1_name]
        date_range = pd.date_range(col_1.min(), col_1.max(), freq="d")
        diff = date_range.difference(col_1).to_frame()
        diff.rename(columns={0: col_1_name}, inplace=True)
        rc = rc.append(diff, ignore_index=True)
        rc = rc.sort_values(col_1_name)

    else:
        raise Exception("unsupported zero_fill column type " + col_1_type)

    for col_name in ds.columns:
        type = column_types.get(col_name)
        if type in ("integer", "real"):
            rc[col_name].fillna(0, inplace=True)

    return rc


def rename_columns(ds, map):
    rc = ds.rename(columns=map)
    return rc


def remove_columns(ds, columns):
    cols_to_drop = [c for c in columns if c in ds]
    rc = ds.drop(columns=cols_to_drop, errors="ignore")
    return rc


def reorder_columns(ds, columns):
    rc = ds[
        [o for o in columns if o in ds.columns]
        + [c for c in ds.columns if c not in columns]
    ]
    return rc


def group_by(ds, columns):
    ordered = ds.columns

    # get the columns to be grouped
    grouped_cols = [c for c in ordered if c not in columns.keys()]

    agg = {}
    rename = {}

    for name, action in columns.items():
        if name not in ds.columns:
            continue

        if action == "COUNT_DISTINCT":
            rename[name] = f"COUNT(DISTINCT {name})"
        else:
            rename[name] = f"{action}({name})"

        if action == "MIN":
            method = "min"
        elif action == "MAX":
            method = "max"
        elif action == "MEDIAN":
            method = "median"
        elif action == "AVG":
            method = "mean"
        elif action == "COUNT":
            method = "count"
        elif action == "SUM":
            method = "sum"
        elif action == "COUNT_DISTINCT":
            method = "nunique"
        elif action == "GROUP_CONCAT":
            agg[name] = lambda x: ", ".join(x)
            continue
        agg[name] = method

    rc = ds
    if grouped_cols:
        rc = ds.groupby(grouped_cols, as_index=False, dropna=False)
    rc = rc.agg(agg)
    if type(rc) == pd.Series:
        rc = rc.to_frame().transpose()
    rc = rc[ordered]
    rc.rename(columns=rename, inplace=True)

    return rc


def histogram_buckets(ds, col, aggregation, bucket_type, custom_buckets):
    if aggregation != "COUNT":
        raise Exception("We only support COUNT aggregations in histograms")

    if bucket_type != "custom_buckets":
        raise Exception("We only support custom_buckets in histograms")
    
    # add the maximum value to the end of the buckets
    max_value = ds[col].max()
    custom_buckets.append(max_value)

    bins = pd.IntervalIndex.from_breaks(custom_buckets, closed='left')
    
    binned = pd.cut(ds[col], bins=bins)
    
    # rename the buckets appropriately
    categories = {}
    for bin in bins:
      categories[bin] = f"{bin.left}-{bin.right - 1}"
    last = bins[-1]
    categories[last] = f"{last.left}-{last.right}"
    binned = binned.cat.rename_categories(categories)
    
    rc = binned.groupby(binned).size()
    
    # add the values before the bins
    before = len(ds[ds[col] < custom_buckets[0]])
    rc[0] += before
    
    # add the maximum values
    max_count = len(ds[ds[col] == max_value])
    rc[-1] += max_count
    
    rc = rc.to_frame().rename_axis(0)
    rc.reset_index(inplace=True)
    rc.rename(columns={0: "Bucket", rc.columns[1]: "Count"}, inplace=True)
    rc['Bucket'] = rc['Bucket'].astype("string")
    return rc


def filter(ds, filters, match_type="all", mode="include"):
    # filter definition
    """
    {
        "column": "col_name",
        "operator": "<=",
        "operand": "op",
        "operand_type": "COLUMN", # or "LITERAL"
    }
    """
    comparisons = []

    for f in filters:
        column = f["column"]
        operator = f["operator"]
        operand = f.get("operand")
        operand_type = f.get("operand_type", "LITERAL")

        if operand_type == "COLUMN":
            operand = ds[operand]
        else:
            if operator and operator[0] in ("<", ">"):
                # attempt to coerce the operand to a number
                try:
                    operand = float(operand)
                except:
                    pass

            if operator == "IN":
                comparisons.append(ds[column].isin(operand))
                continue

        if operator == "IS NULL":
            comparisons.append(ds[column].isnull())
            continue

        if operator == "IS NOT NULL":
            comparisons.append(ds[column].notnull())
            continue

        fn = {
            "=": "eq",
            "!=": "ne",
            ">": "gt",
            ">=": "ge",
            "<": "lt",
            "<=": "le",
        }
        if operator in fn:
            func = getattr(ds[column], fn[operator])
            comparisons.append(func(operand))
            continue

        raise Exception(f"filter operator {operator}")

    rc = comparisons[0]
    for comparison in comparisons[1:]:
        if match_type == "any":
            rc |= comparison
        else:
            rc &= comparison

    if mode == "exclude":
        rc = ~rc

    rc = ds[rc]

    return rc


def sort(ds, columns):
    m = {1: True, -1: False}
    sort_columns = [c["col_name"] for c in columns]
    sort_directions = [m[c["direction"]] for c in columns]
    rc = ds.sort_values(
        sort_columns, ascending=sort_directions, na_position="first", ignore_index=True
    )
    rc = rc.reindex(
        columns=rc.columns
    )

    return rc


def pivot(ds, aggregations):
    def sort_column(column_values):
        """
        Used to sort a list of values based on the order they appeared
        in the previous step. This is to work around the fact that
        pivot_table() automatically sorts the output.
        """
        sort_keys = []
        column_name = ds.columns[0]
        for value in column_values:
            # Find the original index of this key, and use that as the
            # sort key.
            original_index = ds[ds[column_name] == value].index[0]
            sort_keys.append(original_index)

        return sort_keys

    aggfuncs = {"SUM": np.sum, "AVG": np.average, "MAX": np.max}

    if len(aggregations) == 1:
        rc = ds.pivot_table(
            index=ds.columns[0],
            columns=ds.columns[1],
            values=ds.columns[2],
            aggfunc=aggfuncs.get(aggregations[0]),
        )
        rc = rc.fillna(0)
        rc.reset_index(level=0, inplace=True)
        rc.columns.name = ""
    else:
        ordered_cols = list(ds.columns)
        aggfunc = {}

        for i, fn in enumerate(aggregations):
            if len(ordered_cols) <= i + 2:
                break
            col = ordered_cols[i + 2]
            aggfunc[col] = aggfuncs.get(fn)

        rc = ds.pivot_table(index=ordered_cols[0:2], aggfunc=aggfunc)
        rc = rc.unstack()
        rc = rc.reindex(
            columns=rc.columns.reindex(ordered_cols, level=rc.index.nlevels - 1)[0]
        )
        cols = []
        for col in rc.columns.values:
            key = col[1]
            if isinstance(key, float) and key.is_integer():
                key = int(key)
            cols.append(f"{key}:{col[0]}")
        rc.columns = cols
        rc = rc.fillna(0)
        rc.reset_index(level=0, inplace=True)
        rc.columns.name = ""

    # Sort values based on original order, rather than re-ordering
    # which is what pivot_table() does by default.
    if len(rc):
        rc.sort_values(ds.columns[0], key=sort_column, inplace=True, ignore_index=True)

    return rc


def _join(join_type, datasets, join_on_first_n_columns, sort_after_join=True):
    rc = datasets[0]

    ordered_cols = rc.columns[:join_on_first_n_columns]

    for i in range(1, len(datasets)):
        left_data = rc
        right_data = datasets[i]

        # Rename join columns on right-hand dataset to match left-hand names
        renamed_columns = {}
        for col_num in range(0, join_on_first_n_columns):
            renamed_columns[right_data.columns[col_num]] = left_data.columns[col_num]
        right_data_renamed = right_data.rename(columns=renamed_columns)

        rc = left_data.merge(
            right_data_renamed,
            how=join_type,
            left_on=list(left_data.columns[:join_on_first_n_columns]),
            right_on=list(right_data_renamed.columns[:join_on_first_n_columns]),
            suffixes=(None, ":1"),
        )

        # reorder columns
        rc = reorder_columns(rc, ordered_cols)

    if sort_after_join:
        # sort first n columns
        columns = []
        for n in range(join_on_first_n_columns):
            columns.append({"col_name": rc.columns[n], "direction": 1})
        rc = sort(rc, columns)

    return rc


def _column_types_match(datasets, first_n_columns):
    # TODO: handle more than 2 datasets
    if len(datasets) > 2:
        return True

    d1 = datasets[0]
    d2 = datasets[1]
    for n in range(first_n_columns):
        c1 = d1.columns[n]
        c2 = d2.columns[n]
        if d1.dtypes[c1] != d2.dtypes[c2]:
            return False

    return True


def full_outer_join(datasets, join_on_first_n_columns):
    return _join("outer", datasets, join_on_first_n_columns)


def inner_join(datasets, join_on_first_n_columns):
    return _join("inner", datasets, join_on_first_n_columns)


def left_join(datasets, join_on_first_n_columns, sort_after_join=True):
    if not _column_types_match(datasets, join_on_first_n_columns):
        return datasets[0]
    return _join(
        "left", datasets, join_on_first_n_columns, sort_after_join=sort_after_join
    )


def wide_table(ds, column_types: dict = None, column_precision: dict = None):
    """
    Used to display a table of data. This is identical to table(), but
    the table isn't quite as aesthetic. This helps if the table is particularly
    wide since Plotly's tables squish the columns too much to be legible.
    """

    formatters = {}
    for col_name in ds.columns:
        col_type = column_types.get(col_name)
        precision = column_precision.get(col_name)

        if col_type == "percentage":
            precision = precision or 0
            formatters[col_name] = lambda x: f"{{:,.{precision}%}}".format(x)
        elif col_type == "integer":
            formatters[col_name] = lambda x: f"{{:,}}".format(x)
        elif col_type == "currency":
            precision = precision or 2
            formatters[col_name] = lambda x: f"${{:,.{precision}f}}".format(x)
        elif col_type == "real":
            precision = precision or 2
            formatters[col_name] = lambda x: f"{{:,.{precision}f}}".format(x)
        else:
            formatters[col_name] = lambda x: str(x)

    rc = ds.to_html(
        escape=True, notebook=True, index=False, justify="left", formatters=formatters
    )
    display(HTML(rc))


def table(ds, column_types: dict = None, column_precision: dict = None):
    """
    Displays a table of data.

    ds: Dataset to display

    column_types: dict of the type of each column, where the key is the column
                  name and the value is the type. Types can be:

                    - percentage
                    - integer
                    - currency

    column_precision: Precision of each column. Key is the column name and
                        the value is the number of decimal places.
    """

    # Formatting is done using d3 format specifiers:
    # https://github.com/d3/d3-format/blob/main/README.md
    # Here is a tool to help test formats:
    # http://bl.ocks.org/zanarmstrong/05c1e95bf7aa16c4768e

    column_types = column_types or {}
    column_precision = column_precision or {}
    formats = []

    for col_name in ds.columns:

        col_type = column_types.get(col_name)
        precision = column_precision.get(col_name)

        if col_type == "percentage":
            precision = precision or 0
            formats.append(f",.{precision}%")
        elif col_type == "integer":
            formats.append(",.0f")
        elif col_type == "currency":
            precision = precision or 2
            formats.append(f"$,.{precision}f")
        elif col_type == "real":
            precision = precision or 2
            formats.append(f",.{precision}f")
        # elif col_type == "date":
        #    ds[col_name] = pd.to_datetime(ds[col_name], format="%b %d, %Y")
        #    formats.append(None)
        else:
            formats.append(None)

    # ensure nulls render as empty
    ds.fillna("", inplace=True)

    ds = ds.applymap(lambda c: html.escape(c, quote=False) if isinstance(c, str) else c)

    # TODO: https://dash.plotly.com/datatable/width#horizontal-scroll
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=list(ds.columns), align="left"),
                cells=dict(
                    values=[ds[c] for c in ds.columns], format=formats, align="left"
                ),
            )
        ],
    )
    # leave margin for scroll bar
    fig.update_layout(margin=dict(r=25, l=10, t=0, b=0))

    fig.show()


def line(ds):
    """
    Generate a line chart from dataset. Assumes first column
    is the x axis. Any subsequent columns are new datasets.
    """

    # Does not work consistently with multiple lines
    # fig = px.line(ds, x=ds.columns[0], y=ds.columns[1:])

    # ds = sort(ds, [{"col_name": ds.columns[0], "direction": 1}])

    fig = go.Figure()
    for col in ds.columns[1:]:
        fig.add_trace(
            go.Scatter(x=ds[ds.columns[0]], y=ds[col], mode="lines", name=col)
        )

    fig.update_layout(margin=dict(r=10, l=10, t=0, b=0))
    fig.update_yaxes(rangemode="tozero")
    fig.show()


def bar(ds, stacked=False, xaxis_type=None):
    """
    Generate a bar chart from dataset. Assumes first column
    is the x axis. Any subsequent columns are new datasets.

    stacked: Display bars stacked on top of each other rather than
             side by side
    """

    barmode = "group"
    if stacked:
        barmode = "stack"

    fig = px.bar(ds, x=ds.columns[0], y=ds.columns[1:], barmode=barmode)
    fig.update_layout(margin=dict(r=10, l=10, t=0, b=0))
    if xaxis_type:
      fig.update_layout(xaxis={'type': xaxis_type})
    fig.show()


def single_value(ds):
    print(ds.iloc[0, 0])


def pie(ds, max_items=10):
    """
    Generate a pie chart from the dataset. Assumes the dataset has 2 columns,
    the first one being the name and second is the value.

    https://plotly.com/python/pie-charts/
    """

    sorted_data = sort(ds, [{"col_name": ds.columns[1], "direction": -1}])
    if len(sorted_data) > max_items:
        sorted_data.loc[max_items:, sorted_data.columns[0]] = "Other"

    fig = px.pie(
        sorted_data, values=sorted_data.columns[1], names=sorted_data.columns[0]
    )
    fig.update_layout(margin=dict(r=10, l=10, t=0, b=0))
    fig.show()


def area(ds):
    """
    Generate a stacked area chart from dataset. Assumes first column
    is the x axis. Any subsequent columns are new datasets.

    https://plotly.com/python/filled-area-plots/
    """

    fig = go.Figure()
    for col in ds.columns[1:]:
        fig.add_trace(
            go.Scatter(
                x=ds[ds.columns[0]], y=ds[col], mode="lines", stackgroup="one", name=col
            )
        )

    # This does not work consistently:
    # fig = px.area(ds, x=ds.columns[0], y=ds.columns[1:])

    fig.update_layout(margin=dict(r=10, l=10, t=0, b=0))
    fig.show()


def bar_line(ds, last_x_columns_as_lines: int):
    """
    Generate a combination bar and line chart. The first
    column is the x axis. The remaining columns are stacked
    bar charts, and the final x columns are line charts.
    """

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for c in range(1, len(ds.columns) - last_x_columns_as_lines):
        fig.add_trace(
            go.Bar(
                x=ds[ds.columns[0]],
                y=ds[ds.columns[c]],
                offsetgroup=0,
                name=ds.columns[c],
            ),
            secondary_y=False,
        )

    for c in range(len(ds.columns) - last_x_columns_as_lines, len(ds.columns)):
        fig.add_trace(
            go.Scatter(x=ds[ds.columns[0]], y=ds[ds.columns[c]], name=ds.columns[c]),
            secondary_y=True,
        )

    fig.update_layout(margin=dict(r=10, l=10, t=0, b=0))
    fig.show()


def funnel(ds):
    """
    Generate a funnel chart. The first column is the y axis.
    The second column is the x axis.
    """

    fig = px.funnel(ds, x=ds.columns[1], y=ds.columns[0])
    fig.show()


def bubble_map(ds, map_type=None):
    """
    Show a geographical map of data. Assumes the following columns. This is from
    Chartio:

    1. label
    2. latitude
    3. longitude
    4. value (optional)
    5. group (optional)

    Columns 2 through 4 must be numeric,
    column 2 must have values between -90 and 90,
    column 3 must have values between -180 and 180, and
    column 4 can’t be negative.

    https://plotly.com/python/bubble-maps/
    """

    # This column is used for the size fo the bubble
    size = None
    if len(ds.columns) >= 4:
        size = ds[ds.columns[3]]

    # Where is our map focused?
    if map_type == "us":
        scope = "usa"
    elif map_type == "world_map":
        scope = "world"
    else:
        scope = "world"

    fig = px.scatter_geo(
        ds,
        lat=ds[ds.columns[1]],
        lon=ds[ds.columns[2]],
        hover_name=ds[ds.columns[0]],
        size=size,
        scope=scope,
    )

    fig.update_layout(margin=dict(r=10, l=10, t=0, b=0))
    fig.show()
