import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.datasets import make_circles, make_moons, make_blobs, fetch_olivetti_faces, load_breast_cancer
import torch
from torch.utils.data import DataLoader, TensorDataset
import scipy.sparse
import scipy.io
import matplotlib.pyplot as plt
import idx2numpy
from mnist import MNIST


SHUFFLE = True 
IMG_SIZE = 28

folder_path = './Datasets/'

def process_data_labels(data, labels):
    data = np.array(data)
    labels = np.array(labels)
    
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    return data.astype("float"), labels.astype("int")

def get_3d_spheres_np(plot_is_enabled=False):
    num_samples = 500
    centers = [[0, 0, 0], [5, 0, 0]]
    cluster_std = [1.0, 1.0]
    data, labels = make_blobs(n_samples=num_samples, centers=centers, cluster_std=cluster_std, random_state=0)
    
    if(plot_is_enabled):
        # Plot blobs in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        ax.scatter(x, y, z, c=labels, cmap='viridis')
        
        # Set stable viewpoint (adjust as needed)
        ax.view_init(azim=30, elev=20)

        # Adjust perspective (optional - 'ortho' reduces distortion)
        ax.set_proj_type('ortho')
        
        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    
    return data, labels

def get_moons_np():
    data, labels = make_moons(n_samples=1_000, noise=0.05, random_state=0)
    
    return process_data_labels(data, labels)
    
def get_squeezed_gauss_np():
    data, labels = make_blobs(n_samples=1000,random_state=0, centers=[(0.45, 0.5), (0.55, 0.5)], cluster_std=((0.005, 0.15), (0.005, 0.15)))
    
    return process_data_labels(data, labels)
    
def get_gauss_densities_np():
    big_blob, big_blob_label = make_blobs(n_samples=500,random_state=0, centers=[(0, 0)], cluster_std=[(0.25, 1.5)])
    small_x = 3.5
    small_y = 2.5
    small_blobs, small_blobs_labels = make_blobs(n_samples=100,random_state=0, centers=[(small_x, small_y), (small_x, -small_y)], cluster_std=((0.15, 0.15), (0.15, 0.15)))
    small_blobs_labels += 1

    data = np.vstack((big_blob, small_blobs))
    labels = np.concatenate((big_blob_label, small_blobs_labels))
    
    return process_data_labels(data, labels)

def get_australian_np():
    column_names = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "label"]

    df = pd.read_csv(folder_path + "australian/australian.dat", delimiter=" ", header=None, names=column_names)
    data = df[["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14"]]
    labels = df["label"]

    return process_data_labels(data, labels)

def get_wine_np():
    column_names = ["label", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13"]
    df = pd.read_csv(folder_path + "wine/wine.data", header=None, names=column_names)
    data = df[["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13"]]
    labels = df["label"]

    return process_data_labels(data, labels)

def get_ring_np():
    data, labels = make_circles(n_samples=1_000, factor=0.3, noise=0.05, random_state=0)

    return process_data_labels(data, labels)

def get_iris_np():
    columnn_names = ["f1", "f2", "f3", "f4", "label"]
    df = pd.read_csv(folder_path + "iris/iris.data", header=None, names=columnn_names)
    data = df[["f1", "f2", "f3", "f4"]]
    labels = df["label"]

    return process_data_labels(data, labels)

def get_ecoli_np():
    column_names = ["id", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "label"]

    df = pd.read_csv(folder_path + "ecoli/ecoli.data", delimiter="\s+", header=None, names=column_names)
    data = df[["f1", "f2", "f3", "f4", "f5", "f6", "f7"]]
    labels = df["label"]

    return process_data_labels(data, labels)
    
def get_dermatology_np():
    na_column = 33
    labels_column = 34
    df = pd.read_csv(folder_path + "dermatology/dermatology.data", delimiter=",", header=None, na_values="?")
    mean_value = df[na_column].mean()
    df[na_column].fillna(value=mean_value, inplace=True)

    labels = df[labels_column]
    data = df.drop(columns=[labels_column])

    return process_data_labels(data, labels)
    
def get_tcga_np():
    df_data = pd.read_csv(folder_path + 'tcga/data.csv', index_col=0)
    df_labels = pd.read_csv(folder_path + 'tcga/labels.csv', index_col=0)

    labels = np.squeeze(np.array(df_labels))

    return process_data_labels(df_data, labels)

def get_pendigits_np():
    data = np.vstack([
        np.loadtxt(folder_path + "pendigits/pendigits.tra", delimiter=','),
        np.loadtxt(folder_path + "pendigits/pendigits.tes", delimiter=',')
    ])

    pendigits, labels = data[:, :-1], data[:, -1:]
    labels = np.squeeze(labels)
    
    return process_data_labels(pendigits, labels)

def get_hars_np():
    df_train = pd.read_csv(folder_path + 'har/train.csv', index_col=0)
    df_test = pd.read_csv(folder_path + 'har/test.csv', index_col=0)
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    del df_train, df_test
    
    labels = df['Activity']
    df.drop(columns=['subject','Activity'], inplace=True)
    labels = np.squeeze(np.array(labels))

    return process_data_labels(df, labels)   
    
def get_10x73k_np():
    # Original Labels are within 0 to 9. But proper label mapping is required as there are 8 classes.
    data_path = folder_path + "10x_73k/sub_set-720.mtx"
    labels_path = folder_path + "10x_73k/labels.txt"

    # Read data
    data = scipy.io.mmread(data_path)
    data = data.toarray()
    data = np.float32(data)

    # Read labels
    labels = np.loadtxt(labels_path).astype(int)
    labels = LabelEncoder().fit_transform(labels)

    # Scale data
    data = np.log2(data + 1)
    scale = np.max(data)
    data = data / scale

    return process_data_labels(data, labels)

def get_r15_np():
    path = folder_path+"r15/r15.arff"
    df = pd.read_csv(path, skiprows=10, names=['x', 'y', 'class'])
    labels = df["class"].to_numpy()
    labels = LabelEncoder().fit_transform(labels)
    df.drop(columns=["class"], inplace=True)
    data = df.to_numpy()

    return process_data_labels(data, labels)

def get_r3_np():
    # Define the means and covariances for the Gaussian blobs
    means = np.array([[0, 0], [5, 0], [10, 0]])
    sigma = 0.1
    
    covariances = np.array([[[sigma, 0], 
                            [0, sigma]],
                            [[sigma, 0], 
                            [0, sigma]],
                            [[sigma, 0], 
                            [0, sigma]]])
    
    # Number of samples in each cluster
    n_samples = 500
    
    # Generate data for each cluster
    data = []
    for i in range(len(means)):
        cluster_data = np.random.multivariate_normal(means[i], covariances[i], n_samples)
        data.append(cluster_data)
    
    # Combine data from all clusters
    data = np.vstack(data)
    labels = np.repeat(np.arange(len(means)), n_samples)

    return process_data_labels(data, labels)

def get_synthetic_np():
    np.random.seed(42)
 
    # Elements for high projections (DCN) 
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    W = np.random.normal(loc=0.0, scale=1.0, size=(2, 10))
    U = np.random.normal(loc=0.0, scale=1.0, size=(10, 100))
     
    # Number of samples and dimensions
    n_samples = 2500
    n_features = 2
     
    center_1 = np.full(shape=(1, n_features), fill_value=0)
    center_2 = np.full(shape=(1, n_features), fill_value=0)
    center_3 = np.full(shape=(1, n_features), fill_value=0)
    center_4 = np.full(shape=(1, n_features), fill_value=0)
     
    center_2[0][0] = 6
    center_3[0][1] = 6
     
    center_4[0][0] = 6
    center_4[0][1] = 6
     
    # Create data for each Gaussian cluster
    cluster1 = make_blobs(n_samples=n_samples, n_features=n_features, centers=center_1, cluster_std=1.0)
    cluster2 = make_blobs(n_samples=n_samples, n_features=n_features, centers=center_2, cluster_std=1.0)
    cluster3 = make_blobs(n_samples=n_samples, n_features=n_features, centers=center_3, cluster_std=1.0)
    cluster4 = make_blobs(n_samples=n_samples, n_features=n_features, centers=center_4, cluster_std=1.0)
     
    # Combine the clusters to form a 2x2 grid
    data = np.vstack([cluster1[0], cluster2[0], cluster3[0], cluster4[0]])
    data = sigmoid(sigmoid(data @ W) @ U)
    
    labels = np.hstack([np.zeros(n_samples), np.ones(n_samples), 2 * np.ones(n_samples), 3 * np.ones(n_samples)])

    return process_data_labels(data, labels)

def get_emnist_general_np(option, min_index, max_index):
    # Options: balanced, byclass, bymerge, digits, letters, mnist
    mndata = MNIST(folder_path + 'emnist')
    if option == 'mnist':
        mndata.select_emnist('mnist')
    else:    
        mndata.select_emnist('balanced') 
    data, labels = mndata.load_training()
    data_ts, labels_ts = mndata.load_testing()

    data = np.vstack((data, data_ts))
    labels = np.hstack((labels, labels_ts))

    data, labels = process_data_labels(data, labels)
    
    data = np.reshape(data, (-1, 1, IMG_SIZE, IMG_SIZE))
    
    valid_indices = np.where((labels >= min_index ) & (labels < max_index))[0]
    
    return data[valid_indices], labels[valid_indices]

def get_emnist_balanced_digits_np():
    return get_emnist_general_np('digits', 0, 10)

def get_emnist_mnist_np():
    return get_emnist_general_np('mnist', 0, 10)
    
def get_emnist_balanced_letters_A_J_np():
    return get_emnist_general_np('letters', 10, 20)

def get_emnist_balanced_letters_K_T_np():
    return get_emnist_general_np('letters', 20, 30)

def get_emnist_balanced_letters_U_Z_np():
    return get_emnist_general_np('letters', 30, 36)

def get_waveform_v1_np():
    df_data = pd.read_csv(folder_path + 'waveform_v1/data.csv', header = None)
    df_labels = pd.read_csv(folder_path + 'waveform_v1/labels.csv', header = None)
 
    return process_data_labels(df_data, df_labels)

def get_kmnist_np():
    train_data = np.load(folder_path + "kmnist/kmnist-train-imgs.npz")['arr_0']
    test_data = np.load(folder_path + "kmnist/kmnist-test-imgs.npz")['arr_0']
    train_labels = np.load(folder_path + "kmnist/kmnist-train-labels.npz")['arr_0']
    test_labels = np.load(folder_path + "kmnist/kmnist-test-labels.npz")['arr_0']

    data = np.vstack((train_data, test_data))
    labels = np.hstack((train_labels, test_labels))

    data = np.reshape(data, (-1, IMG_SIZE * IMG_SIZE))
    
    data, labels = process_data_labels(data, labels)

    data = np.reshape(data, (-1, 1, IMG_SIZE, IMG_SIZE))

    return data, labels

def get_fashionmnist_np():
    train_images = idx2numpy.convert_from_file(folder_path + 'fashionmnist/train-images-idx3-ubyte').astype(np.float32)
    test_images = idx2numpy.convert_from_file(folder_path + 'fashionmnist/t10k-images-idx3-ubyte').astype(np.float32)
    train_labels = idx2numpy.convert_from_file(folder_path + 'fashionmnist/train-labels-idx1-ubyte')
    test_labels = idx2numpy.convert_from_file(folder_path + 'fashionmnist/t10k-labels-idx1-ubyte')

    data = np.vstack((train_images, test_images))
    labels = np.hstack((train_labels, test_labels))

    data = np.reshape(data, (-1, IMG_SIZE * IMG_SIZE))
    
    data, labels = process_data_labels(data, labels)

    data = np.reshape(data, (-1, 1, IMG_SIZE, IMG_SIZE))

    return data, labels

def get_olivetti_faces_np():
    faces = fetch_olivetti_faces()
    
    return process_data_labels(faces.data, faces.target)


def get_breast_cancer_np():
    breast_cancer = load_breast_cancer()
    
    return process_data_labels(breast_cancer.data, breast_cancer.target)

def get_dataset(dataset_name, batch_size=64):
    function_name = "get_" + dataset_name + "_np"
    function_to_call = globals()[function_name]
    
    data_np, labels_np = function_to_call()
   
    # Convert to tensor dataset
    data = torch.Tensor(data_np)
    data_shape = data.shape[1]
    labels = torch.Tensor(labels_np)
    final_dataset = TensorDataset(data, labels)
    dataloader = DataLoader(final_dataset, batch_size=batch_size, shuffle=SHUFFLE)

    return dataloader, data_shape, data_np, labels_np
