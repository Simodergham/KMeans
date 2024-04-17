import numpy as np
from random import sample
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class KMeansCustom:
    def __init__(self, num_clusters=3, max_iterations=100):
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.cluster_centers = []
    
    def initialize_centers(self, data):
        self.cluster_centers = sample(list(data), self.num_clusters)
    
    def assign_clusters(self, data):
        clusters = [[] for _ in range(self.num_clusters)]
        for point in data:
            distances = [np.linalg.norm(point - center) for center in self.cluster_centers]
            closest_center = distances.index(min(distances))
            clusters[closest_center].append(point)
        return clusters
    
    def update_centers(self, clusters):
        for i in range(self.num_clusters):
            if clusters[i]:
                self.cluster_centers[i] = np.mean(clusters[i], axis=0)
    
    def fit(self, data):
        self.initialize_centers(data)
        
        for _ in range(self.max_iterations):
            clusters = self.assign_clusters(data)
            prev_centers = self.cluster_centers.copy()
            self.update_centers(clusters)
            
            if all(np.array_equal(prev, curr) for prev, curr in zip(prev_centers, self.cluster_centers)):
                break
    
    def predict(self, data):
        predictions = []
        for point in data:
            distances = [np.linalg.norm(point - center) for center in self.cluster_centers]
            closest_center = distances.index(min(distances))
            predictions.append(closest_center)
        return predictions



data, true_labels = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)


plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=true_labels, cmap='viridis', marker='o', edgecolor='k')
plt.title('Visualisation de la Distribution des Données')
plt.xlabel('Caractéristique 1')
plt.ylabel('Caractéristique 2')
plt.colorbar(label='Étiquette de Cluster (Vraie)')
plt.show()


kmeans_custom = KMeansCustom(num_clusters=4)
kmeans_custom.fit(data)


clusters = kmeans_custom.predict(data)

plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis')
centroids = np.array(kmeans_custom.cluster_centers)
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='x')
plt.title('Points de Données et Centroïdes')
plt.xlabel('Caractéristique 1')
plt.ylabel('Caractéristique 2')
plt.show()