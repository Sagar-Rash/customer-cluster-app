# Customer-Cluster-App
Determines cluster which a customer belongs in on a scatter plot given age, income, and spending score.
## Features
Streamlit application with input form for Age, Income, and Spending Score.

Prediction of customer cluster position (helps with determining what to target them with).

Accessible via Streamlit Community Cloud: https://customer-cluster-app-sr.streamlit.app/

# Dataset
The dataset is the Mall Customer dataset which has multiple copies available on Kaggle. The main features used in this model are

Age

Annual Income (in Thousands of Dollars)

Spending Score (1-100)

# Technologies Used
Streamlit: For the application.

Scikit-learn: For model training and evaluation.

Pandas and NumPy: For data preprocessing and manipulation.

Matplotlib and Seaborn: For exploratory data analysis and visualization.

Plotly: Plotting 3-Dimensional scatter plot of customer data (clusters)

# Model
The predictive model was trained using a Mall Customer dataset which has multiple copies available on Kaggle.

# Installation (to local computer):
If you want to run the application locally, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/your-username/customer-cluster-app.git
cd customer-cluster-app
```
2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\\Scripts\\activate`
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the Streamlit application:
```bash
streamlit run Cust_Cluster_app_streamlit.py
```
# Note that some filepaths may need to be changed based on the local system (Windows vs. Linux) (backslash vs. forward slash)
