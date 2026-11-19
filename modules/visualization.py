
import plotly.express as px


# 1. Create a histogram
def create_histogram(df, column):

    fig = px.histogram(
        df,
        x=column,
        title=f"Distribution of {column}",
        template="plotly_dark"
    )

    return fig


# 2. Create a bar chart
def create_bar_chart(df, column):

    counts = df[column].value_counts().reset_index()

    counts.columns = [column, "Count"]

    fig = px.bar(
        counts,
        x=column,
        y="Count",
        title=f"Distribution of {column}",
        template="plotly_dark"
    )

    return fig


# 3. Create a correlation heatmap
def create_heatmap(df):

    numerical_df = df.select_dtypes(include="number")

    correlation = numerical_df.corr()

    fig = px.imshow(
        correlation,
        text_auto=True,
        title="Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        template="plotly_dark"
    )

    return fig
