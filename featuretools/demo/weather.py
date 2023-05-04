import pandas as pd

import featuretools as ft


def load_weather(nrows=None, return_single_table=False):
    """
    Load the Australian daily-min-temperatures weather dataset.

    Args:

        nrows (int): Passed to nrows in ``pd.read_csv``.
        return_single_table (bool): Exit the function early and return a dataframe.

    """
    filename = "daily-min-temperatures.csv"
    print("Downloading data ...")
    url = f"https://api.featurelabs.com/datasets/{filename}?library=featuretools&version={ft.__version__}"
    data = pd.read_csv(url, index_col=None, nrows=nrows)
    return data if return_single_table else make_es(data)


def make_es(data):
    es = ft.EntitySet("Weather Data")

    es.add_dataframe(
        data,
        dataframe_name="temperatures",
        index="id",
        make_index=True,
        time_index="Date",
    )
    return es
