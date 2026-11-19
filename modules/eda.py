
import pandas as pd


# 1. Generate descriptive statistics
def generate_descriptive_statistics(df):

    numerical_df = df.select_dtypes(include="number")

    statistics = numerical_df.describe()

    return statistics
# 2. Calculate correlations between numerical columns

def generate_correlation_matrix(df):

    numerical_df = df.select_dtypes(include="number")

    correlation_matrix = numerical_df.corr()

    return correlation_matrix
# 3. Analyze categorical columns

def generate_categorical_statistics(df):

    categorical_df = df.select_dtypes(
        include=["object", "category"]
    )

    statistics = categorical_df.describe().T

    return statistics