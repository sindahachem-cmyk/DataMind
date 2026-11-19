
import pandas as pd


def load_data(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):

        # Automatically detect the CSV separator
        df = pd.read_csv(
            uploaded_file,
            sep=None,
            engine="python"
        )

    elif file_name.endswith(".xlsx"):

        df = pd.read_excel(uploaded_file)

    else:
        raise ValueError("Unsupported file format")

    return df

