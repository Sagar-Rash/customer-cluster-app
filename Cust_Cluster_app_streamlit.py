from pandas import read_csv
from seaborn import heatmap
from matplotlib.pyplot import subplots
import pickle
import streamlit as st
from plotly.express import scatter_3d
from plotly.graph_objects import Scatter3d

st.title("Cluster Solution App")
st.write("""
This app will determine the cluster that a customer (given customer information) is part of on a scatter plot.
It uses a K-means cluster model.
""")



# Prepare the form
with st.form("user_inputs"):
    st.subheader("Select Input Parameters")
    

    Age = st.number_input("Age", 
                            min_value=18, 
                            step=1)
    Annual_Income = st.number_input("Annual Income (in Thousands of Dollars)", 
                            min_value=10, 
                            step=1)
    Spending_Score = st.number_input("Spending Score (1-100)", 
                            min_value=1, max_value=100,
                            step=1)
    submitted = st.form_submit_button("Determine Customer's Cluster:")
        
    if submitted:
        
        prediction_input = [[Age, Annual_Income, Spending_Score]]
        cluster_pickle = open(r"models/Cluster_Model_k6.pkl", "rb")
        cluster_model = pickle.load(cluster_pickle)
        cluster_pickle.close()    
        cluster_df = read_csv(r"data/3d_cluster_df.csv")
        centers_df = read_csv(r"data/Cluster_centers_3d.csv")
        
        new_prediction = cluster_model.predict(prediction_input)
            
        st.subheader("Prediction Result:")
        st.write(f"The customer is predicted to be in cluster {new_prediction[0]}.")

        
        corr_df = read_csv(r"data/mall_customers.csv")
        

        st.subheader("Figures and Data")
        # Correlation plot
        st.write("The correlation plot is shown below:")
        corrs = corr_df.corr(numeric_only=True,method='spearman') 
        fig1, ax1 = subplots(figsize=(6, 6))
        heatmap(corrs, annot=True,ax=ax1)
        st.pyplot(fig1)
        
        
        st.write("The count of customers in each cluster can be seen below:")
        st.dataframe(cluster_df['Cluster'].value_counts().sort_values())
        
        st.write("The 3D plot of the data is below, with the customer and cluster centers highlighted:")
        
        fig2 = scatter_3d(cluster_df, x="Age", y="Annual_Income", z="Spending_Score", color='Cluster')
        fig2.add_trace(Scatter3d(
                x=[Age],
                y=[Annual_Income],
                z=[Spending_Score],
                mode='markers',
                marker=dict(size=8, color='red', symbol='x'),
                textposition='top center',
                name='Customer'
            ))
        # Cluster center
        fig2.add_trace(Scatter3d(
                x=centers_df["0"],
                y=centers_df["1"],
                z=centers_df["2"],
                mode='markers',
                marker=dict(size=8, color='green', symbol='diamond'),
                textposition='top center',
                name='Cluster center'
            ))
        # Show the interactive plot
        st.plotly_chart(fig2)
