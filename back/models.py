# ---------------------------------- Imports --------------------------------

import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
import os
import uvicorn
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from fastapi import FastAPI
import base64
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
import matplotlib


matplotlib.use('Agg')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ---------------------------------- Data Processing --------------------------------


def create_img_folder():
    # Get the current directory
    current_directory = os.getcwd()

    # Define the path to the img folder
    img_folder_path = os.path.join(current_directory, "img")

    # Check if the img folder exists
    if not os.path.exists(img_folder_path):
        # If it doesn't exist, create it
        os.makedirs(img_folder_path)
        print("img folder created successfully.")
    else:
        print("img folder already exists.")



# The data process set X on : 'Income', 'Score' and Y on : 'Gender'.

df = pd.read_csv(r'./data/Mall_Customers.csv')
df.rename(index=str, columns={'Annual Income (k$)': 'Income',
                              'Spending Score (1-100)': 'Score'}, inplace=True)

X = df.drop(['CustomerID', 'Gender'], axis=1)

# Feature normalization
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("Data Processing validated")


# ---------------------------------- Kmeans API --------------------------------
app = FastAPI()
random_state =42

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.post('/prediction_kmeans')
async def prediction_kmeans(n_clusters : int):
    model = KMeans
    kmeans = model(n_clusters=n_clusters, init='k-means++', random_state=random_state)
    kmeans.fit(X_scaled)
    # Calculate silhouette score
    silhouette = silhouette_score(X_scaled, kmeans.labels_)
    print(f"Silhouette Score: {silhouette}")
    # Calculate Davies–Bouldin index
    davies_bouldin = davies_bouldin_score(X_scaled, kmeans.labels_)
    print(f"Davies–Bouldin Index: {davies_bouldin}")
    # Plot function
    plt.figure(figsize=(12, 8))
    for cluster_label in range(n_clusters):  # Loop through each cluster label
        cluster_points = X[kmeans.labels_ == cluster_label]
        centroid = cluster_points.mean(axis=0)  # Calculate the centroid as the mean position of the data points
        plt.scatter(cluster_points['Income'], cluster_points['Score'],
                    s=50, label=f'Cluster {cluster_label + 1}')  # Plot points for the current cluster
        plt.scatter(centroid[0], centroid[1], s=300, c='black', marker='*', label=f'Centroid {cluster_label + 1}')  # Plot the centroid
        plt.title('Clusters of Customers')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        # Save plot to BytesIO buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        # Encode plot as base64 string
        plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plot_file = f"img/plot_kmeans_{n_clusters}_clusters.png"
        plt.savefig(plot_file)
        print("Plot validated")
    print("Plot and Metrics for Kmeans sended")
    print(f"Kmeans Score, davies_bouldin : {davies_bouldin}, silhouette : {silhouette} effectued with {n_clusters} clusters.")
    # Prepare response
    response = {
        'model_name': 'Kmeans',
        'n_clusters': n_clusters,
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'plot_image': plot_base64
    }
    return response


# ---------------------------------- Agglom API --------------------------------




@app.post('/prediction_agglo')
async def prediction_agglo(n_clusters : int):
    model = AgglomerativeClustering
    agglom = model(n_clusters, linkage='average').fit(X)
    X['Labels'] = agglom.labels_
    # davies_bouldin metric
    silhouette = silhouette_score(X[['Income', 'Score']], agglom.labels_)
    print(f"Silhouette Score: {silhouette}")
    # davies_bouldin metric
    davies_bouldin = davies_bouldin_score(X[['Income', 'Score']], agglom.labels_)
    print(f"Davies–Bouldin Index: {davies_bouldin}")
    # Plot function
    # Plot the clusters
    plt.figure(figsize=(12, 8))
    for cluster_label in range(n_clusters):  
        # Extract points belonging to the current cluster
        cluster_points = X[X['Labels'] == cluster_label]
        # Calculate the centroid as the mean position of the data points in the cluster
        centroid = cluster_points[['Income', 'Score']].mean(axis=0)
        # Plot points for the current cluster
        plt.scatter(cluster_points['Income'], cluster_points['Score'],
                    s=50, label=f'Cluster {cluster_label + 1}')
        # Plot the centroid
        plt.scatter(centroid[0], centroid[1], s=300, c='black', marker='*', label=f'Centroid {cluster_label + 1}')
        plt.title('Clusters of Customers')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        # Save plot to BytesIO buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        # Encode plot as base64 string
        plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plot_file = f"img/plot_agglo_{n_clusters}_clusters.png"
        plt.savefig(plot_file)
        print("Plot validated")
        
    print("Plot and Metrics for Agglom sended")
    print(f"Agglomeration Model with {n_clusters}, Davies Bouldin Score: {davies_bouldin}, Silhouette Score : {silhouette}.")
    # Prepare response
    response = {
        'model_name': 'Agglomeration',
        'n_clusters': n_clusters,
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'plot_image': plot_base64
    }
    return response
    
# ---------------------------------- Server --------------------------------


if __name__ == '__main__':
    create_img_folder()
    uvicorn.run(app, host='0.0.0.0', port=8001)