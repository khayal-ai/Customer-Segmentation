import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from kmeans_model import run_kmeans


df=pd.read_csv("data/store_customers.csv")
#understand our dataset
print(df.head())

print(df.dtypes)
print(df.info())

print(df.isnull().sum()) #we notice all attributes but customerId contain null values 

#Handle missing values

#For Gender use Mode imputation since it is categroical
df['Gender']= df['Gender'].fillna(df['Gender'].mode()[0])

#For Numerical values use Median imputation -> its more ROBUST to outliers
df['Age']= df['Age'].fillna(df['Age'].median())
df['Annual Income (k$)']= df['Annual Income (k$)'].fillna(df['Annual Income (k$)'].median())
df['Spending Score (1-100)']= df['Spending Score (1-100)'].fillna(df['Spending Score (1-100)'].median())

print(df.isnull().sum())

#Encode Gender 
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])

# Save dataset BEFORE removing outliers
df_before = df.copy()

# IQR OUTLIER REMOVAL
cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

for c in cols:
    Q1 = df[c].quantile(0.25)
    Q3 = df[c].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[c] >= lower) & (df[c] <= upper)]

print("Shape after removing outliers:", df.shape)

# Visualization code (your code)
cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

for c in cols:

    plt.figure(figsize=(10,4))

    # Before removing outliers
    plt.subplot(1,2,1)
    sns.boxplot(y=df_before[c].dropna())
    plt.title(f"{c} Before")

    # After removing outliers
    plt.subplot(1,2,2)
    sns.boxplot(y=df[c].dropna())
    plt.title(f"{c} After")

    plt.tight_layout()
    plt.show()


scaler = MinMaxScaler()

num_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

df_minmax = df.copy()
df_minmax[num_cols] = scaler.fit_transform(df_minmax[num_cols])

print("\nMinMax Normalized Data:")
print(df_minmax.head())


scaler = StandardScaler()
df_zscore = df.copy()
df_zscore[num_cols] = scaler.fit_transform(df_zscore[num_cols])

print("\nZ-score Standardized Data:")
print(df_zscore.head())

plt.figure(figsize=(6,5))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


pca = PCA(n_components=2)

X_pca = pca.fit_transform(df[num_cols])

print("PCA Shape:", X_pca.shape)

# Run KMeans from external file
df = run_kmeans(df, df_minmax)

# Average characteristics of each cluster
cluster_summary = df.groupby('Cluster')[['Age','Annual Income (k$)','Spending Score (1-100)']].mean()

print("\nCluster Summary:")
print(cluster_summary)

# Number of customers in each cluster
print("\nNumber of customers in each cluster:")
print(df['Cluster'].value_counts())

# Visualization of number of customers per cluster
plt.figure(figsize=(6,4))
sns.countplot(x=df['Cluster'], palette='Set2')
plt.title("Number of Customers per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.show()

# More detailed cluster visualization
plt.figure(figsize=(8,6))

sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    hue=df['Cluster'],
    palette='Set1',
    s=80
)

plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()