
import streamlit as st
import pandas as pd
import numpy as np

from modules.data_loader import load_data

from modules.data_cleaning import (
    analyze_data_quality,
    analyze_missing_values,
    remove_duplicates,
    handle_missing_values
)

from modules.eda import (
    generate_descriptive_statistics,
    generate_correlation_matrix,
    generate_categorical_statistics
)

from modules.visualization import (
    create_histogram,
    create_bar_chart,
    create_heatmap
)

from modules.machine_learning import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest
)


# --------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="DataMind | Data Science Studio",
    page_icon="📊",
    layout="wide"
)


# --------------------------------
# 2. CUSTOM DESIGN
# --------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: #FFFFFF;
}

[data-testid="stSidebar"] {
    background-color: #161B22;
}

h1, h2, h3 {
    color: #FFFFFF;
}

.stMetric {
    background-color: #1C222B;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# 3. SIDEBAR
# --------------------------------

st.sidebar.title("📊 DataMind")

st.sidebar.markdown(
    "### Your AI Data Science Studio"
)

st.sidebar.markdown("""
Upload, clean, explore, visualize, 
and train machine learning models 
on your datasets.
""")

st.sidebar.divider()

st.sidebar.markdown("**Tools**")
st.sidebar.markdown("""
- Data Upload
- Data Cleaning
- Statistics
- Exploratory Data Analysis
- Visualization
- Machine Learning
""")


# --------------------------------
# 4. MAIN HEADER
# --------------------------------

st.title("📊 DataMind")
st.subheader("Your AI Data Science Studio")

st.markdown("""
Analyze your datasets, discover insights,
create visualizations, and train machine learning models.
""")

st.divider()


# --------------------------------
# 5. DATA UPLOAD
# --------------------------------

st.header("📂 Data Upload")

uploaded_file = st.file_uploader(
    "Upload your CSV or Excel file",
    type=["csv", "xlsx"]
)


if uploaded_file is not None:

    try:

        df = load_data(uploaded_file)

        st.success("Dataset loaded successfully!")

        # --------------------------------
        # 6. DATA OVERVIEW
        # --------------------------------

        st.header("📋 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        with col3:
            st.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )

        with col4:
            st.metric(
                "Duplicate Rows",
                int(df.duplicated().sum())
            )

        st.subheader("Data Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        st.subheader("Column Information")

        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Non-Null Values": df.notnull().sum().values
        })

        st.dataframe(
            info_df,
            use_container_width=True
        )


        # --------------------------------
        # 7. DATA QUALITY
        # --------------------------------

        st.header("🧹 Data Quality")

        quality_report = analyze_data_quality(df)

        q1, q2, q3, q4 = st.columns(4)

        with q1:
            st.metric(
                "Total Rows",
                quality_report["total_rows"]
            )

        with q2:
            st.metric(
                "Total Columns",
                quality_report["total_columns"]
            )

        with q3:
            st.metric(
                "Missing Values",
                quality_report["missing_values"]
            )

        with q4:
            st.metric(
                "Duplicate Rows",
                quality_report["duplicate_rows"]
            )

        st.subheader("Missing Values per Column")

        missing_report = analyze_missing_values(df)

        st.dataframe(
            missing_report,
            use_container_width=True
        )


        # --------------------------------
        # 8. DATA CLEANING
        # --------------------------------

        st.header("🛠️ Data Cleaning")

        st.subheader("Remove Duplicate Rows")

        if st.button("Remove Duplicates"):

            cleaned_df, removed_rows = remove_duplicates(df)

            st.success(
                f"{removed_rows} duplicate rows removed."
            )

            st.write("Cleaned dataset:")

            st.dataframe(
                cleaned_df.head(10),
                use_container_width=True
            )

            st.download_button(
                "Download Cleaned Data",
                cleaned_df.to_csv(index=False).encode("utf-8"),
                "cleaned_data.csv",
                "text/csv"
            )


        st.subheader("Handle Missing Values")

        cleaning_method = st.selectbox(
            "Choose a cleaning method",
            [
                "Remove rows",
                "Fill with mean",
                "Fill with median"
            ]
        )

        if st.button("Clean Missing Values"):

            cleaned_df = handle_missing_values(
                df,
                cleaning_method
            )

            st.success("Missing values processed!")

            st.dataframe(
                cleaned_df.head(10),
                use_container_width=True
            )

            st.download_button(
                "Download Cleaned Dataset",
                cleaned_df.to_csv(index=False).encode("utf-8"),
                "missing_values_cleaned.csv",
                "text/csv"
            )


        # --------------------------------
        # 9. EXPLORATORY DATA ANALYSIS
        # --------------------------------

        st.header("🔎 Exploratory Data Analysis")

        st.subheader("Descriptive Statistics")

        descriptive_stats = generate_descriptive_statistics(df)

        st.dataframe(
            descriptive_stats,
            use_container_width=True
        )


        st.subheader("Categorical Statistics")

        categorical_stats = generate_categorical_statistics(df)

        if not categorical_stats.empty:

            st.dataframe(
                categorical_stats,
                use_container_width=True
            )

        else:

            st.info("No categorical columns found.")


        st.subheader("Correlation Matrix")

        correlation_matrix = generate_correlation_matrix(df)

        if not correlation_matrix.empty:

            st.dataframe(
                correlation_matrix,
                use_container_width=True
            )

        else:

            st.info("No numerical columns available for correlation.")


        # --------------------------------
        # 10. VISUALIZATION
        # --------------------------------

        st.header("📈 Data Visualization")

        all_columns = df.columns.tolist()

        numerical_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        categorical_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()


        if numerical_columns:

            st.subheader("Histogram")

            histogram_column = st.selectbox(
                "Select a numerical column",
                numerical_columns,
                key="histogram_column"
            )

            histogram_fig = create_histogram(
                df,
                histogram_column
            )

            st.plotly_chart(
                histogram_fig,
                use_container_width=True
            )


        if categorical_columns:

            st.subheader("Bar Chart")

            bar_column = st.selectbox(
                "Select a categorical column",
                categorical_columns,
                key="bar_column"
            )

            bar_fig = create_bar_chart(
                df,
                bar_column
            )

            st.plotly_chart(
                bar_fig,
                use_container_width=True
            )


        if len(numerical_columns) >= 2:

            st.subheader("Correlation Heatmap")

            heatmap_fig = create_heatmap(df)

            st.plotly_chart(
                heatmap_fig,
                use_container_width=True
            )


        # --------------------------------
        # 11. MACHINE LEARNING
        # --------------------------------

        st.header("🤖 Machine Learning")

        st.markdown(
            "Train machine learning models "
            "to predict numerical values."
        )


        if len(numerical_columns) >= 2:

            # Choose algorithm
            algorithm = st.selectbox(
                "Choose Machine Learning Algorithm",
                [
                    "Linear Regression",
                    "Decision Tree Regression",
                    "Random Forest Regression"
                ]
            )


            # Choose target column
            target_column = st.selectbox(
                "Choose the column you want to predict",
                numerical_columns,
                key="target_column"
            )


            # Choose feature columns
            available_features = [
                col for col in numerical_columns
                if col != target_column
            ]

            feature_columns = st.multiselect(
                "Choose input features",
                available_features,
                default=available_features[:2]
            )


            if st.button("Train Model"):

                if len(feature_columns) == 0:

                    st.warning(
                        "Please select at least one input feature."
                    )

                else:

                    try:

                        # Select the correct model
                        if algorithm == "Linear Regression":

                            model_function = train_linear_regression

                        elif algorithm == "Decision Tree Regression":

                            model_function = train_decision_tree

                        else:

                            model_function = train_random_forest


                        # Train the model
                        model, metrics, predictions, y_test = (
                            model_function(
                                df,
                                target_column,
                                feature_columns
                            )
                        )


                        st.success(
                            f"{algorithm} trained successfully!"
                        )


                        # Model performance
                        st.subheader("📊 Model Performance")

                        metric1, metric2, metric3 = st.columns(3)

                        with metric1:
                            st.metric(
                                "MAE",
                                round(metrics["MAE"], 3)
                            )

                        with metric2:
                            st.metric(
                                "MSE",
                                round(metrics["MSE"], 3)
                            )

                        with metric3:
                            st.metric(
                                "R² Score",
                                round(metrics["R2 Score"], 3)
                            )


                        # Actual vs Predicted
                        st.subheader(
                            "Actual vs Predicted Values"
                        )

                        results_df = pd.DataFrame({
                            "Actual Values": y_test.values,
                            "Predicted Values": predictions
                        })

                        st.dataframe(
                            results_df,
                            use_container_width=True
                        )


                    except Exception as e:

                        st.error(
                            f"Error training model: {e}"
                        )

        else:

            st.info(
                "At least two numerical columns are required "
                "to train a machine learning model."
            )


        # --------------------------------
        # 12. FOOTER
        # --------------------------------

        st.divider()

        st.markdown("""
        <div style="text-align:center; color:gray;">
            <p>DataMind | AI Data Science Studio</p>
            <p>Built with Python, Streamlit, Pandas & Scikit-learn</p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:

        st.error(f"Error loading dataset: {e}")


else:

    st.info(
        "👈 Upload a dataset from the upload section to get started."
    )
