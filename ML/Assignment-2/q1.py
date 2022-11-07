import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI

col_names = ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Species"]

df = pd.read_csv("iris.csv",names = col_names)

#print(df.head())

#data preprocessing

label_map = {
    'Iris-setosa':0,
    'Iris-versicolor':1,
    'Iris-virginica':2
}

df['Species'].replace(label_map,inplace = True)

#print(df.head())

#print(df.isnull().any())

X = df.drop(['Species'],axis = 1)
Y = df['Species']

#print(X.head())

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

#print(X[:5])

# Importing pca model and creating the pipeline

pca = PCA()
X_pca = pca.fit_transform(X)
cov_mat = pca.get_covariance()
print("\ncovariance matrix : ")
print(cov_mat)

relative_cov = pca.explained_variance_ratio_
print("\nrelative  covariance  matrix :")
print(relative_cov)

print("\nTotal variance with two components : ",np.sum(relative_cov[:2])/np.sum(relative_cov))
print("95% variance was preserved so num_components = 2.\n")
num_components = 2
pca = PCA(num_components)
X_pca = pca.fit_transform(X)
princ_df = pd.DataFrame(data = X_pca, columns = ['principal component 1', 'principal component 2'])
xpca = pca.transform(X)
names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
new_df = pd.concat([princ_df,df[['Species']]],axis=1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1, 2]
colors = ['g', 'b', 'r']
for target, color in zip(targets,colors):
    indicesToKeep = new_df['Species'] == target
    #print(indicesToKeep)
    ax.scatter(new_df.loc[indicesToKeep, 'principal component 1'], new_df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
    ax.legend(names)
    ax.grid()
plt.show()
# KNN with features extracted from pca
data = xpca
k = np.arange(2,9,dtype = int)
#print(k)
num_iter = 30
epsilon = 1e-12
nmi = []
for i in k:
    random_indices = np.random.choice(data.shape[0], size=i, replace=False)
    cluster_centroids = data[random_indices,:]
    #print(cluster_centroids)
    errors = []
    distance_from_centroids = np.zeros((data.shape[0],i+2))
    # Cluster assignment
    for j in range(num_iter):
        for p in range(data.shape[0]):
            for q in range(i):
                #finding distance to all cluster_centroids
                distance_from_centroids[p,q] = np.linalg.norm(data[p,:]-cluster_centroids[q,:])
            minimum_dist = np.min(distance_from_centroids[p,:i])
            centroid = int((np.where(distance_from_centroids[p,:i] == minimum_dist)[0]))
            distance_from_centroids[p,i] = centroid
            distance_from_centroids[p,i+1] = minimum_dist

        # Re-estimation of cluster positions
        for p in range(i):
            ind = np.where(distance_from_centroids[:,i] == p)
            cluster_centroids[p,:] = np.mean(data[ind,:][0],axis=0)

        # Objective Function to be minimized
        errors.append(np.mean(distance_from_centroids[:,i+1]))

        if j > 5:
            if abs(errors[j]-errors[j-1]) < epsilon:
                break

    pred_data = distance_from_centroids[:,i]
    nmi.append(NMI(Y,pred_data))

#print(nmi)

#print(pca)

plt.plot(k,nmi)
plt.xlabel('K')
plt.ylabel('NMI')
plt.title('k vs NMI')
plt.show()

#print(errors)