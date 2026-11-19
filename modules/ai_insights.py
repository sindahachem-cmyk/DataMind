
import pandas as pd


def generate_ai_insights(df):

    insights = []

    # 1. Dataset overview
    rows, columns = df.shape

    insights.append(
        f"Dataset contains {rows} rows and {columns} columns."
    )

    # 2. Data quality
    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())

    insights.append(
        f"Missing values: {missing}."
    )

    insights.append(
        f"Duplicate rows: {duplicates}."
    )

    # 3. Remove ID columns from numerical analysis
    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    useful_columns = [
        col for col in numeric_columns
        if "id" not in col.lower()
    ]

    # 4. Numerical insights
    for column in useful_columns:

        average = df[column].mean()

        insights.append(
            f"Average {column}: {average:.2f}."
        )

    # 5. Revenue analysis
    if "Total Amount" in df.columns:

        total_revenue = df["Total Amount"].sum()

        average_transaction = df["Total Amount"].mean()

        insights.append(
            f"Total revenue: {total_revenue:,.2f}."
        )

        insights.append(
            f"Average transaction value: "
            f"{average_transaction:.2f}."
        )

        # 6. Product category analysis
        if "Product Category" in df.columns:

            category_revenue = df.groupby(
                "Product Category"
            )["Total Amount"].sum().sort_values(
                ascending=False
            )

            if not category_revenue.empty:

                best_category = category_revenue.idxmax()
                worst_category = category_revenue.idxmin()

                insights.append(
                    f"Highest revenue category: "
                    f"{best_category} "
                    f"({category_revenue.max():,.2f})."
                )

                insights.append(
                    f"Lowest revenue category: "
                    f"{worst_category} "
                    f"({category_revenue.min():,.2f})."
                )

                insights.append(
                    f"Recommendation: Focus marketing efforts "
                    f"on {best_category}, while investigating "
                    f"ways to improve {worst_category} sales."
                )

    # 7. Customer analysis
    if "Age" in df.columns:

        average_age = df["Age"].mean()

        insights.append(
            f"Average customer age: {average_age:.1f} years."
        )
    # 8. Customer spending by gender
    if "Gender" in df.columns and "Total Amount" in df.columns:

        gender_revenue = df.groupby("Gender")["Total Amount"].sum()

        if not gender_revenue.empty:
            best_gender = gender_revenue.idxmax()

            insights.append(
                f"Highest revenue by gender: {best_gender} "
                f"({gender_revenue.max():,.2f})."
            )

    # 9. Customer spending by age group
    if "Age" in df.columns and "Total Amount" in df.columns:

        age_groups = pd.cut(
            df["Age"],
            bins=[0, 25, 35, 45, 60, 120],
            labels=["18-25", "26-35", "36-45", "46-60", "60+"]
        )

        age_revenue = df.groupby(
            age_groups, observed=False
        )["Total Amount"].mean().dropna()

        if not age_revenue.empty:
            best_age_group = age_revenue.idxmax()

            insights.append(
                f"Highest average spending age group: "
                f"{best_age_group} "
                f"({age_revenue.max():.2f})."
            )

    # 10. Average transaction by product category
    if "Product Category" in df.columns and "Total Amount" in df.columns:

        category_average = df.groupby(
            "Product Category"
        )["Total Amount"].mean()

        if not category_average.empty:
            best_category = category_average.idxmax()

            insights.append(
                f"Highest average transaction category: "
                f"{best_category} "
                f"({category_average.max():.2f})."
            )
    # 11. Correlation analysis
    if len(useful_columns) >= 2:

        correlation = df[useful_columns].corr().abs()

        pairs = correlation.where(
            ~correlation.eq(1)
        ).stack()

        if not pairs.empty:

            strongest_pair = pairs.idxmax()
            strength = pairs.max()

            insights.append(
                f"Strongest numerical relationship: "
                f"{strongest_pair[0]} and "
                f"{strongest_pair[1]} "
                f"(correlation: {strength:.2f})."
            )

    return insights
