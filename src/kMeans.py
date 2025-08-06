
# Train K means cluster model
def trainKModel(cluster_df, input_params, n_clusters = 5, print_details = False):
    # Note that the first entry in input_params will be x on the scatter plot and the second will be y
    # if more than 2 input params, plot will be 3D
    from sklearn.cluster import KMeans
    from seaborn import scatterplot
    from matplotlib.pyplot import show
    from plotly.express import scatter_3d

    try:
        # create model and fit to input parameters
        kmodel = KMeans(n_clusters,init="k-means++").fit(cluster_df[input_params])
        # Add cluster labels to dataframe
        cluster_df['Cluster'] = kmodel.labels_
        if print_details:
            # Plots, values, and visualizations
            print('The cluster centers for this model can be seen below:')
            print(kmodel.cluster_centers_)
            print('\n')
            
            print('The first five rows of this dataset (including clusters) can be seen below:')
            print(cluster_df.head())
            print('\n')
            
            print('Number of values per cluster:')
            print(cluster_df['Cluster'].value_counts())
            print('\n')
            
            print('Visualization of clusters:')
            
            if len(input_params) == 2:
                # 2D scatterplot
                scatterplot(x=input_params[0], y = input_params[1], data=cluster_df, hue='Cluster', palette='colorblind')
                show()
            else:

                # Create a 3D scatter plot with plotly and then show it
                fig = scatter_3d(cluster_df, x=input_params[0], y=input_params[1], z=input_params[2], color='Cluster')
                fig.show()
        return kmodel, cluster_df
    except Exception as e:
        print(f"Error occurred: {e}")
        return 0

def get_scores_df(cluster_df, input_params, plot_graphs = False):
    # Get scores and determine optimal K based on Silhouette score
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from pandas import DataFrame
    from matplotlib.pyplot import xlabel, ylabel, title, show
    
    try:
        # Get WCSS and Silhouette for each model from k = 3 to 9
        # Then create dataframe with scores per k
        k = range(3,9)
        K = []
        WCSS = []
        ss = []
        for i in k:
            kmodel = KMeans(n_clusters=i,init="k-means++").fit(cluster_df[input_params])
            wcss_score = kmodel.inertia_
            ypred = kmodel.labels_
            sil_score = silhouette_score(cluster_df[input_params], ypred)
            K.append(i)
            WCSS.append(wcss_score)
            ss.append(sil_score)
        scores_df = DataFrame({'cluster': K, 'WCSS_Score':WCSS, 'Silhouette_Score':ss})
                
        optimal_k_idx = scores_df['Silhouette_Score'].idxmax() # Index of optimal k value (Max Silhouette Score)
        optimal_k = scores_df['cluster'].iloc[optimal_k_idx] # Optimal K value     
        if plot_graphs:  
            # Elbow
            print('The Elbow plot when trying the model with input_params can be seen below:')
            scores_df.plot(x='cluster', y = 'WCSS_Score')
            xlabel('No. of clusters')
            ylabel('WCSS Score')
            title('Elbow Plot')
            show()
            print('\n')
            # Silhouette
            print('The Silhouette Score plot when trying the model with input_params can be seen below:')
            scores_df.plot(x='cluster', y='Silhouette_Score')
            xlabel('No. of clusters')
            ylabel('Silhouette Score')
            title('Silhouette Plot')
            show()
            print('\n')

            print("The scores table can be seen below:")
            print(scores_df)
            print('\n')
            
            print("The optimal K value is " + str(optimal_k))
        
        return scores_df, optimal_k
    except Exception as e:
        print(f"Error occurred: {e}")
        return 0
        


