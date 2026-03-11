import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

def run_kmeans(df, df_minmax):

    # use normalized data for clustering
    X = df_minmax[['Annual Income (k$)', 'Spending Score (1-100)']]

    wcss = []

    for i in range(1,11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1,11), wcss, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.show()

    kmeans = KMeans(n_clusters=5, random_state=42)

    df['Cluster'] = kmeans.fit_predict(X)

    plt.figure(figsize=(8,6))

    sns.scatterplot(
        x=df['Annual Income (k$)'],
        y=df['Spending Score (1-100)'],
        hue=df['Cluster'],
        palette='Set1'
    )

    plt.title("Customer Segmentation using K-Means")
    plt.show()

    return df