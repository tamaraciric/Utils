#!/usr/bin/env python
# coding: utf-8



from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Utils import conn_file
import seaborn as sns
sns.set(style="ticks")

from sklearn.preprocessing import StandardScaler




def show_optimal_k(pivot,minK,maxK):
    
    test_range = range(minK,maxK)
    kmeans = [KMeans(n_clusters=i) for i in test_range]
    score = [kmeans[i].fit(pivot).score(pivot) for i in range(len(kmeans))]
    
    #plot elbow curve
    plt.plot( score,test_range, 'bx-')
    plt.xlabel('score')
    plt.ylabel('number of clusters')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    

def _kmeans(data,k):
    return KMeans(n_clusters=k, random_state=40, init='k-means++', n_init=10, max_iter=300)


def get_kmeans_param(data,k,features):
    
    kmeans_k = _kmeans(data,k)
    cluster_k = kmeans_k.fit_predict(data)
    centroids_k  = kmeans_k.cluster_centers_
    centroids_df = pd.DataFrame(centroids_k,columns= features.tolist())
    
    centroid_labels = [centroids_k[i] for i in cluster_k]

    
    centroids_df = centroids_df.reset_index()
    centroids_df.columns = ['klaster'] + features.tolist()
    
    return cluster_k,centroids_k,centroids_df
    


#visualization
def show_segments(centroids_df):

    
    df_x = pd.melt(centroids_df,id_vars='klaster')
    df_x.columns = [u'klaster', u'type', u'value']
   

    plt.figure(figsize = (20,5))
    result = df_x.pivot(index='klaster', columns='type', values='value')
    sns.heatmap(result,vmin=0,vmax=30, annot=True, fmt=".2g", linewidths=.5)
    plt.show()
    

def show_dist_per_cluster (data):
    
    trans_dist = []
       
    for i in range(data['klaster'].nunique()):
        trans_dist.append([i,data[data['klaster']==i].shape[0]])
    return trans_dist






