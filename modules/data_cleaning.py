
import pandas as pd


# 1. Analyze overall dataset quality
def analyze_data_quality(df):

    quality_report = {
        "total_rows": df.shape[0],
        "total_columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    return quality_report


# 2. Analyze missing values in each column
def analyze_missing_values(df):

    missing_report = pd.DataFrame({
        "Missing Values": df.isnull().sum(),
        "Missing Percentage": (
            df.isnull().sum() / len(df) * 100
        ).round(2)
    })

    return missing_report


# 3. Remove duplicate rows
def remove_duplicates(df):

    original_rows = len(df)

    cleaned_df = df.drop_duplicates()

    removed_rows = original_rows - len(cleaned_df)

    return cleaned_df, removed_rows


# 4. Handle missing values
def handle_missing_values(df, method):

    cleaned_df = df.copy()

    if method == "Remove rows":

        cleaned_df = cleaned_df.dropna()

    elif method == "Fill with mean":

        numeric_columns = cleaned_df.select_dtypes(
            include="number"
        ).columns

        cleaned_df[numeric_columns] = (
            cleaned_df[numeric_columns].fillna(
                cleaned_df[numeric_columns].mean()
            )
        )

    elif method == "Fill with median":

        numeric_columns = cleaned_df.select_dtypes(
            include="number"
        ).columns

        cleaned_df[numeric_columns] = (
            cleaned_df[numeric_columns].fillna(
                cleaned_df[numeric_columns].median()
            )
        )

    return cleaned_df
