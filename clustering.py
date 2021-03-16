import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

from glob import glob
from tslearn.utils import to_time_series
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans , silhouette_score
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from numpy import savetxt
from scipy.cluster.hierarchy import single, complete, average, ward, dendrogram, fcluster


def hierarchical_clustering(dist_matrix, method='complete'):
    if method == 'complete':
        Z = complete(dist_matrix)
    if method == 'single':
        Z = single(dist_matrix)
    if method == 'average':
        Z = average(dist_matrix)
    if method == 'ward':
        Z = ward(dist_matrix)
    
    fig = plt.figure(figsize=(20, 20))
    dn = dendrogram(Z)
    plt.title(f"Dendrogram for {method}-linkage with correlation distance")
    plt.show()
    
    return Z

def main(args):

    data_dir = './Data/User Categorization/'

    if args.method == 'K':
        print('Working on K-means clustering')
        ts_dataset = [] 

        #Only take the first 500 unique ID's
        n_samples = 500

        for i in range(n_samples):
            csv_file = pd.read_csv(data_dir + str(i) + '.csv')
            time_series_df = csv_file[(~csv_file['f_1'].isnull()) & (~csv_file['f_2'].isnull())]
            time_series_seq = list(time_series_df[['f_1','f_2','f_3']].values)
            ts_dataset.append(time_series_seq)  
        
        #Preparing Time-series dataset
        formatted_dataset = to_time_series_dataset(ts_dataset)

        silhouette_scores = []
        n_clusters = [2,3,4,5,6]


        for cluster in n_clusters:
            km = TimeSeriesKMeans(n_clusters=cluster, metric="dtw",verbose=True,max_iter=5)
            y_pred = km.fit_predict(formatted_dataset)
            s_score = silhouette_score(formatted_dataset, y_pred, metric="dtw")
            silhouette_scores.append(s_score)

        sns.lineplot(x=n_clusters, y=silhouette_scores, sort=False)

        #Optimal clusters
        km = TimeSeriesKMeans(n_clusters=2, metric="dtw",verbose=True,max_iter=5)
        y_pred = km.fit_predict(formatted_dataset)
        df = pd.DataFrame(data=y_pred,columns=['Cluster No.'])
        df.to_csv('./kmeans_clustering.csv',index=False)
        
        #Visualise Clusters
        sz = formatted_dataset.shape[1]
        plt.figure(figsize=(20,20))

        for yi in range(2):
            plt.subplot(3, 3, 2 + yi)
            for xx in formatted_dataset[y_pred == yi]:
                plt.plot(xx.ravel(), "k-", alpha=.2)
            plt.plot(km.cluster_centers_[yi].ravel(), "r-")
            plt.xlim(0, sz)
            plt.ylim(-500000, 500000)
            plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
                    transform=plt.gca().transAxes)
            if yi == 1:
                plt.title("DTW $k$-means")
        plt.tight_layout()
        plt.show()


    if args.method == 'H':
        #Hierarchical clustering
        print('Working on Hierarchical clustering')
        #Build distance matrix
        manual_dist_matrix = True
        n_samples = 500

        if manual_dist_matrix == False:
            distance_matrix = np.zeros(shape=(n_samples,n_samples))

            for i in range(n_samples):
                for j in range(n_samples):
                    sequence_1_df = pd.read_csv('./Data/User Categorization/'+ str(i) + '.csv')
                    sequence_2_df = pd.read_csv('./Data/User Categorization/' + str(j) + '.csv')
                    
                    seq_1 = sequence_1_df[(~sequence_1_df['f_1'].isnull()) & (~sequence_1_df['f_2'].isnull())]
                    seq_2 = sequence_2_df[(~sequence_2_df['f_1'].isnull()) & (~sequence_2_df['f_2'].isnull())]
                    
                    x = seq_1[['f_1','f_2','f_3']].values
                    y = seq_2[['f_1','f_2','f_3']].values

                    distance, path = fastdtw(x, y, dist=euclidean)
                    
                    if i != j:
                        distance_matrix[i,j] = distance


            savetxt('distance_matrix.csv', distance_matrix, delimiter=',')

        distance_matrix  = np.genfromtxt('distance_matrix.csv', delimiter=',')
        linkage_matrix = hierarchical_clustering(distance_matrix)

        # select maximum number of clusters
        cluster_labels = fcluster(linkage_matrix, 4, criterion='maxclust')
        print(np.unique(cluster_labels))

        categorization_df = []
        files_list = os.listdir('./Data/User Categorization')

        for files in files_list:
            csv_file = pd.read_csv('./Data/User Categorization/' + str(files))
            unique_id = files[:-4]
            csv_file['ID'] = unique_id
            categorization_df.append(csv_file)

        df = pd.concat(categorization_df,axis=0,ignore_index=True)

        #filter out null values
        filtered_df = df[(~df['f_1'].isnull()) & (~df['f_2'].isnull())]

        df_vis = filtered_df.sort_values(by='ID')
        df_vis['ID'] = df_vis['ID'].astype('int')
        df_vis = df_vis[df_vis['ID'] <= 499].sort_values(by='ID').reset_index(drop=True)
        df_vis_fil = df_vis.groupby('ID')['f_1','f_2','f_3'].mean().reset_index()
        df_vis_fil['Cluster'] = cluster_labels 
        df_vis_fil.to_csv('./hier_clustering.csv',index=False)

        #Plotting Visualisation 3D scatterplot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = np.array(df_vis_fil['f_1'])
        y = np.array(df_vis_fil['f_2'])
        z = np.array(df_vis_fil['f_3'])

        ax.scatter(x,y,z, marker="s", c=df_vis_fil["Cluster"], cmap="RdBu")

        plt.show()

    else:
        print('Please input K or H clustering method correctly')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, metavar='', help = 'Clustering Method = [K-means,Hierarchical]')
    args = parser.parse_args()
    main(args)

