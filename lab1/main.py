import matplotlib.pyplot as plt

import input
import clusters
import quality

# get input data
data = input.getInput('lab01.csv')

# show input data
plt.scatter(data[:,0], data[:,1], marker='x', color='purple')

plt.title('Input data')
plt.show()

# train ms_model
clusters.trainModel(data)

# get cluster centers
labels, cluster_centers, num_clusters = clusters.getClusters()

# show cluster centers
print('Cluster centers = ', cluster_centers)

markers = '*+sv.p^>'
plt.figure()

for i, marker in zip(range(num_clusters), markers):
    plt.scatter(data[labels==i, 0], data[labels==i, 1], marker=marker)

    cluster_center = cluster_centers[i]
    plt.plot(cluster_center[0], cluster_center[1], marker = 'o', markeredgecolor = 'purple', markersize = 15)

plt.title('Clusters')
plt.show()

# get quality
values = range(2, 15)
scores, num_clusters = quality.getQuality(data, values)

# show quality
for i, score in zip(values, scores):
    print('Number of clusters = ', i)
    print('Score = ', score)

print('Optimal cluster number: ', num_clusters)

plt.figure()
plt.bar(values, scores, width = 0.9, color = 'purple', align = 'center')

plt.title('Cluster quality')
plt.show()

# get cluster areas
output, x_vals, y_vals = clusters.getClusterAreas(data)

# show cluster areas
plt.figure()
plt.clf()
plt.imshow(output, interpolation='nearest', extent=(x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()), cmap=plt.cm.Paired, aspect='auto', origin='lower')

for i, marker in zip(range(num_clusters), markers):
    plt.scatter(data[labels==i, 0], data[labels==i, 1], marker = marker)

    cluster_center = cluster_centers[i]
    plt.plot(cluster_center[0], cluster_center[1], marker = 'o', markeredgecolor = 'purple', markersize = 15)

plt.title('Cluster areas (k = {})'.format(num_clusters))
plt.show()