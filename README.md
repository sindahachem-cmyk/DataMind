# 📊 DataMind — AI Data Science Studio

**An interactive data science application built with Python and Streamlit.**

DataMind is a data science studio that allows users to upload datasets, clean data, explore statistics, create visualizations, and train machine learning models through a simple and interactive interface.

## 🚀 Features

### 1. Data Upload
- Upload CSV and Excel files.
- Preview datasets.
- Explore rows, columns, and data types.

### 2. Data Quality & Cleaning
- Detect missing values.
- Identify duplicate rows.
- Remove duplicates.
- Handle missing values using mean, median, or row removal.

### 3. Exploratory Data Analysis (EDA)
- Generate descriptive statistics.
- Analyze categorical variables.
- Explore numerical correlations.

### 4. Data Visualization
- Create histograms.
- Generate bar charts.
- Display correlation heatmaps.

### 5. Machine Learning
Train and evaluate three regression algorithms:

- Linear Regression
- Decision Tree Regression
- Random Forest Regression

Evaluate models using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly

## 📈 Example Results

Tested using the Student Performance dataset to predict final grades (`G3`) from `G1` and `G2`.

| Algorithm | MAE | MSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 1.262 | 4.212 | 0.795 |
| Decision Tree | 1.395 | 4.839 | 0.764 |
| Random Forest | 1.362 | 4.838 | 0.764 |

Linear Regression achieved the best results in this particular experiment.

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/DataMind.git
```

Navigate to the project folder:

```bash
cd DataMind
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal.

## 📁 Project Structure

```text
DataMind/
│
├── app.py
├── modules/
│   ├── ai_insights.py
│   ├── data_cleaning.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── machine_learning.py
│   ├── statistics.py
│   └── visualization.py
│
├── pages/
├── database/
├── results/
├── tests/
│
├── requirements.txt
└── README.md
```

## 🎯 Project Objective

The goal of DataMind is to make essential data science tasks more accessible through an integrated application, combining data preprocessing, exploratory analysis, visualization, and machine learning.

## 👩‍💻 Author

**Sinda Hachem**

Applied Mathematics Student | Data Science

Interested in Artificial Intelligence, Data Science, and Machine Learning.

---

*Built with Python, Streamlit, and a passion for data science.*