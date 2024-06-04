from skimage import morphology, measure, segmentation, feature
from scipy import ndimage as ndi
import numpy as np
import scipy.spatial
import tifffile as tif
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import alphashape
import pandas as pd
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
import os
from patchify import patchify, unpatchify

def calculate_delaunay_triangles(centroids):
    tri = scipy.spatial.Delaunay(centroids)
    return tri.simplices

    
# def measure_diameter_along_line(image, center, normal, modified_raw_image, length):
def measure_diameter_along_line(image, center, normal, length):
    z, y, x = center
    norm = np.linalg.norm(normal)
    if norm == 0:
        return None, None, None

    normal = normal / norm

    z_line = np.linspace(z - length * normal[0], z + length * normal[0], 2 * length)
    y_line = np.linspace(y - length * normal[1], y + length * normal[1], 2 * length)
    x_line = np.linspace(x - length * normal[2], x + length * normal[2], 2 * length)

    coordinates = np.vstack((z_line, y_line, x_line))

    coordinates = coordinates.reshape(3, -1)


    # draw on image
    z_indices = np.round(z_line).astype(int)
    y_indices = np.round(y_line).astype(int)
    x_indices = np.round(x_line).astype(int)

    z_indices = np.clip(z_indices, 0, image.shape[0] - 1)
    y_indices = np.clip(y_indices, 0, image.shape[1] - 1)
    x_indices = np.clip(x_indices, 0, image.shape[2] - 1)

    # modified_raw_image[z_indices, y_indices, x_indices] = 2

    line_profile = ndi.map_coordinates(image, coordinates, order=1)

    inverted_profile = np.max(line_profile) - line_profile

    peaks, _ = scipy.signal.find_peaks(inverted_profile)

    if len(peaks) >= 2:
        lowest_peaks = sorted(peaks, key=lambda x: line_profile[x])[:2]
        diameter = np.abs(lowest_peaks[1] - lowest_peaks[0])
    else:
        diameter = None

    # return z_line, y_line, x_line, diameter, modified_raw_image
    return z_line, y_line, x_line, diameter


# def measure_median_diameter(image, center, normal, modified_raw_image, length):
def measure_median_diameter(image, center, normal, length):
    diameters = []
    normal = normal / np.linalg.norm(normal)
    arbitrary_vector = np.array([1, 0, 0])
    if np.allclose(normal, arbitrary_vector):
        arbitrary_vector = np.array([0, 1, 0])
    if np.allclose(normal, arbitrary_vector):
        arbitrary_vector = np.array([0, 0, 1])
    perpendicular_vector = np.cross(normal, arbitrary_vector)
    unit_normal = perpendicular_vector / np.linalg.norm(perpendicular_vector)

    # new_iamge = np.copy(modified_raw_image)
    
    for i in range(18):
        angle = np.deg2rad(i * 10)
        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)
        one_minus_cos = 1.0 - cos_theta
        ux, uy, uz = normal

        rotation_matrix = np.array([
            [cos_theta + ux**2 * one_minus_cos, ux*uy*one_minus_cos - uz*sin_theta, ux*uz*one_minus_cos + uy*sin_theta],
            [uy*ux*one_minus_cos + uz*sin_theta, cos_theta + uy**2 * one_minus_cos, uy*uz*one_minus_cos - ux*sin_theta],
            [uz*ux*one_minus_cos - uy*sin_theta, uz*uy*one_minus_cos + ux*sin_theta, cos_theta + uz**2 * one_minus_cos]
        ])

        
        rotated_normal = rotation_matrix.dot(unit_normal)        
        # z_line, y_line, x_line, diameter, new_iamge = measure_diameter_along_line(image, center, rotated_normal, new_iamge, length)
        z_line, y_line, x_line, diameter = measure_diameter_along_line(image, center, rotated_normal, length)
        if diameter is not None:
            diameters.append(diameter)


    # return np.median(diameters), new_iamge
    return np.median(diameters)

def process_batch(batch):
    labeled_stack = measure.label(batch, connectivity=1)
    regions = measure.regionprops(labeled_stack)
    centroids = [region.centroid for region in regions]
    return centroids

def cal_diameters(label_name,raw_name,csv_file_path,current_path):

    inputimage = tif.imread(label_name)

    map_name = "centroids_map.npy"

    map_file_path = os.path.join(current_path, map_name)

    centroids_map = []

    if os.path.exists(map_file_path):
        centroids_map = np.load(map_file_path)
        print("File exists. Loaded centroids_map from", map_file_path)
    else:
        # patches = patchify(inputimage, (64, 64, 64), step=64)

        # centroids_num = 0
        # for i in range(patches.shape[0]):
        #     for j in range(patches.shape[1]):
        #         for k in range(patches.shape[2]):
        #             start_x, start_y, start_z = i * 64, j * 64, k * 64
        #             single_patch = patches[i, j, k, :, :, :]
        #             batch_centroids = process_batch(single_patch)
        #             for centroid in batch_centroids:
        #                 abs_x, abs_y, abs_z = centroid
        #                 abs_x += start_x
        #                 abs_y += start_y
        #                 abs_z += start_z
        #                 centroids_map.append((abs_x, abs_y, abs_z))
        #                 centroids_num = centroids_num + 1
        
        # np.save(map_file_path, centroids_map)
        inputimage_shape = inputimage.shape 
        desired_block_count = (2, 2, 2)

        block_size = tuple(max(1, int(inputimage_shape[dim] / desired_block_count[dim])) for dim in range(3))
        print("block_size: ",block_size)
        patches = patchify(inputimage, block_size, step=block_size[0])
        
        centroids_num = 0

        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                for k in range(patches.shape[2]):
                    start_x, start_y, start_z = i * block_size[0], j * block_size[1], k * block_size[2]
                    single_patch = patches[i, j, k, :, :, :]
                    batch_centroids = process_batch(single_patch)
                    for centroid in batch_centroids:
                        abs_x, abs_y, abs_z = centroid
                        abs_x += start_x
                        abs_y += start_y
                        abs_z += start_z
                        centroids_map.append((abs_x, abs_y, abs_z))
                        centroids_num += 1

        np.save(map_file_path, centroids_map)        
        print("File not found. Generated and saved centroids_map to", map_file_path)
                
    print("Number of centroids:", len(centroids_map))


    centroids = np.array(centroids_map)

    dbscan = DBSCAN(eps=30, min_samples=20)
    dbscan.fit(centroids)

    labels = dbscan.labels_
    unique_labels = set(labels) - {-1}
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print(f"Estimated number of clusters: {n_clusters}")

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(centroids[labels == -1, 0], centroids[labels == -1, 1], centroids[labels == -1, 2], c='k', label='Noise')

    
    # for label in unique_labels:
    #     ax.scatter(centroids[labels == label, 0], centroids[labels == label, 1], centroids[labels == label, 2], label=f'Cluster {label}')

    # ax.legend()
    # ax.set_title('DBSCAN Clustering')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(centroids[labels == -1, 0], centroids[labels == -1, 1], centroids[labels == -1, 2], c='k', marker='x', label='Noise')

    if unique_labels:
        largest_cluster_label = max(unique_labels, key=list(labels).count)
        # ax.scatter(centroids[labels == largest_cluster_label, 0], centroids[labels == largest_cluster_label, 1], centroids[labels == largest_cluster_label, 2], s=10, edgecolors='yellow', label=f'Largest Cluster {largest_cluster_label}')
        largest_cluster_x = centroids[labels == largest_cluster_label, 0].tolist()
        largest_cluster_y = centroids[labels == largest_cluster_label, 1].tolist()
        largest_cluster_z = centroids[labels == largest_cluster_label, 2].tolist()
        largest_cluster_centroids_list = list(zip(largest_cluster_x, largest_cluster_y, largest_cluster_z))
        largest_cluster_size = len(largest_cluster_centroids_list)
        print(f"Size of the largest cluster: {largest_cluster_size}")
        # print("Centroids of the largest cluster:", largest_cluster_centroids_list[:5])


    # ax.legend()
    # ax.set_title('DBSCAN: Largest Cluster and Noise')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()


    def find_nearest_triangles(centroids, tri_centers, k):
        nearest_triangles = [[] for _ in centroids]

        for j, centroid in enumerate(centroids):
            for i, tri_center in enumerate(tri_centers):
                distance = np.linalg.norm(centroid - tri_center)
                nearest_triangles[j].append((i, distance))

        nearest_triangles = [sorted(tris, key=lambda x: x[1])[:k] for tris in nearest_triangles]

        return nearest_triangles

    def compute_normals_for_centroids(alpha_shape, nearest_triangles):
        centroid_normals = []
        vertices = alpha_shape.vertices

        for near_tris in nearest_triangles:
            normal = np.array([0.0, 0.0, 0.0])
            for tri_idx, _ in near_tris:
                tri = alpha_shape.faces[tri_idx]
                tri_points = np.array([vertices[int(idx)] for idx in tri])
                vector1 = tri_points[1] - tri_points[0]
                vector2 = tri_points[2] - tri_points[0]
                normal += np.cross(vector1, vector2)
            normal /= np.linalg.norm(normal) if np.linalg.norm(normal) != 0 else 1
            centroid_normals.append(normal)

        return centroid_normals

    centroids_map = largest_cluster_centroids_list

    alpha = 0.002
    k = 1
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(*zip(*centroids_map))
    # plt.show()
    alpha_shape = alphashape.alphashape(centroids_map, alpha)
    triangles = alpha_shape.faces


    # print(len(triangles))
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot_trisurf(*zip(*alpha_shape.vertices), triangles=alpha_shape.faces)
    # plt.show()
    normals = []

    raw_img = tif.imread(raw_name)
    triangles = alpha_shape.faces
    vertices = alpha_shape.vertices
    tri_center = []


    def triangle_area(vertices, face):
        p1, p2, p3 = vertices[face]
        a = np.linalg.norm(p1 - p2)
        b = np.linalg.norm(p2 - p3)
        c = np.linalg.norm(p3 - p1)
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        return area

    areas = np.array([triangle_area(vertices, face) for face in triangles])

    print("triangle number: ", len(triangles))
    print(areas[:100])

    # plt.hist(areas, bins=10000, color='#0504aa', alpha=0.7, rwidth=0.85)
    # plt.title('Triangle Areas Histogram')
    # plt.xlabel('Area')
    # plt.ylabel('Frequency')
    # plt.grid(axis='y', alpha=0.75)
    # plt.savefig(current_path + "/Triangle Areas Histogram.png")

    plt.hist(areas, bins=1000, range=(0, 100), color='#0504aa', alpha=0.7, rwidth=0.85)
    plt.title('Triangle Areas Histogram')
    plt.xlabel('Area')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(current_path + "/Triangle_Areas_Histogram.png")

    return

    for i, tri in enumerate(triangles):
        tri_points = [vertices[int(idx)] for idx in tri]
        tri_center.append(np.mean(tri_points, axis=0))

    # modified_raw_image = None
    n = 1000 
    median_diameters = []
    for i in range(0, len(largest_cluster_centroids_list), n):
        batch_centroids = largest_cluster_centroids_list[i:i+n]

        nearest_triangles = find_nearest_triangles(batch_centroids, tri_center, k)
        centroid_normals = compute_normals_for_centroids(alpha_shape, nearest_triangles)
        normals = centroid_normals

        for center, normal in zip(reversed(batch_centroids), reversed(normals)):
            median_diameter = measure_median_diameter(raw_img, center, normal, 20)
            # print("Median Diameter:", median_diameter)
            median_diameters.append(median_diameter)
        df = pd.DataFrame(median_diameters, columns=["Median Diameter"])
        df.to_csv(csv_file_path, index=False)


    # nearest_triangles = find_nearest_triangles(centroids_map, triangles, k)
    # centroid_normals = compute_normals_for_centroids(centroids_map, triangles, nearest_triangles)
    # normals = centroid_normals

    # median_diameters = []

    # for center, normal in zip(reversed(centroids), reversed(normals)):
    #     median_diameter = measure_median_diameter(raw_img, center, normal, 20)
    #     # median_diameter, current = measure_median_diameter(raw_img, center, normal, modified_raw_image, 20)
    #     # modified_raw_image = current
    #     print("Median Diameter:", median_diameter)
    #     median_diameters.append(median_diameter)
        # if median_diameter > 9 and median_diameter < 15:
        #     modified_raw_image = current
        # if median_diameter > 15:
        #     modified_raw_image = current
        # if median_diameter < 9:
        #     modified_raw_image = current
            # print(median_diameter)
        
            # print("Median Diameter:", median_diameter)
            # median_diameters.append(median_diameter)
        

    median_diameters_data = median_diameters
    mean_median_diameter = sum(median_diameters_data) / len(median_diameters_data)
    median_median_diameter = sorted(median_diameters_data)[len(median_diameters_data) // 2]
    std_dev_median_diameter = (sum((x - mean_median_diameter) ** 2 for x in median_diameters_data) / len(median_diameters_data)) ** 0.5
    min_median_diameter = min(median_diameters_data)
    max_median_diameter = max(median_diameters_data)

    statistics = {
        "Mean": mean_median_diameter,
        "Median": median_median_diameter,
        "Standard Deviation": std_dev_median_diameter,
        "Minimum": min_median_diameter,
        "Maximum": max_median_diameter
    }

    print(statistics)

    plt.hist(median_diameters_data, bins=5, color='skyblue', edgecolor='black')
    plt.title('Distribution of Median Diameters')
    plt.xlabel('Median Diameter')
    plt.ylabel('Frequency')
    plt.savefig(current_path + "/median_diameters_distribution.png")
    # plt.show()

    df = pd.DataFrame(median_diameters, columns=["Median Diameter"])
    df.to_csv(csv_file_path, index=False)
    # modified_output_file = input_path + raw_name + fr"modified_raw_image.tif"
    # tif.imsave(modified_output_file, modified_raw_image)


csv_file_path = r"/mnt/research-data/test_prediction/Standardized Crops 2024/method_1_RhebNeuron2_old/RhebNeuron2_median_diameters.csv"

label_name = "/mnt/research-data/test_prediction/Standardized Crops 2024/method_1_RhebNeuron2_old/clear_10000_RhebNeuron2_conbine_10000.tif"

raw_name = "/mnt/research-data/raw_data/RhebNeuron/RhebNeuron2_Raw.tif"

current_path = "/mnt/research-data/test_prediction/Standardized Crops 2024/method_1_RhebNeuron2_old"

cal_diameters(label_name,raw_name,csv_file_path,current_path)