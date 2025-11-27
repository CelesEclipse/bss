from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA


def normalize(v):
    v = v - v.min()
    return v / (v.max() + 1e-12)


def show_image_from_vector(vec, size=(256, 256), title=""):
    img = vec.reshape(size)
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')


def load_and_prepare(path_or_file, size=(256, 256)):
    """Load image from a file path or file-like object, convert to grayscale, resize and flatten.

    Returns (arr_2d, flat_vector).
    """
    # load img
    img = Image.open(path_or_file).convert('L')

    # resize
    img = img.resize(size)

    # convert to numpy matrix
    arr = np.asarray(img, dtype=np.float32) / 255.0

    # flatten into 1D vector
    flat = arr.flatten()

    return arr, flat


def separate_images(img_files, size=(256, 256)):
    """Run ICA on N already-mixed input images and return the separated sources.

    Parameters
    ----------
    img_files: list or tuple of path or file-like
        A sequence of mixed observations that PIL can open. Must contain at least 2 items.
    size: tuple[int, int]
        Target size for resizing and reconstruction.

    Returns
    -------
    dict with recovered PIL.Image objects in grayscale mode ("L"). Keys are
    'recovered_1', 'recovered_2', ..., 'recovered_N'.
    """
    # Validate input
    if not isinstance(img_files, (list, tuple)):
        raise TypeError("img_files must be a list or tuple of paths or file-like objects")
    m = len(img_files)
    if m < 2:
        raise ValueError("At least two mixed images are required")

    # Prepare data from the mixed images
    flats = []
    for f in img_files:
        _, flat = load_and_prepare(f, size=size)
        flats.append(flat)

    # Stack into matrix X of observations, shape (n_pixels, m)
    X = np.column_stack(flats)

    # Center and whiten, then ICA to estimate sources
    Xc = X - X.mean(axis=0)
    pca = PCA(whiten=True, random_state=0)
    Xw = pca.fit_transform(Xc)
    ica = FastICA(n_components=m, whiten=False, random_state=0)
    S_est = ica.fit_transform(Xw)

    def vector_to_image(vec):
        v = normalize(vec)
        img_arr = (v * 255.0).reshape(size).astype(np.uint8)
        return Image.fromarray(img_arr, mode="L")

    recovered = [vector_to_image(S_est[:, i]) for i in range(m)]

    return {f"recovered_{i+1}": img for i, img in enumerate(recovered)}


if __name__ == "__main__":
    # Keep a simple CLI run using the original example images
    img1_arr, img1_flat = load_and_prepare("src_6.png")
    img2_arr, img2_flat = load_and_prepare("src_5.PNG")
    img3_arr, img3_flat = load_and_prepare("src_4.PNG")

    print("Image 1 shape:", img1_arr.shape)
    print("Flatten shape:", img1_flat.shape)

    # stack into one mixed image S
    # shape = (N, 2)
    S = np.column_stack((img1_flat, img2_flat, img3_flat))
    print("S shape: ", S.shape)

    # Create mixing matrix
    A = np.array([
        [1.0, 0.5, 2],
        [0.4, 1.0, 1],
        [0.5, 0.2, 1]
    ])

    # Generate the mixed signals: X = Sdet(A)
    X = S @ A.T
    print("X shape:", X.shape)

    show_image_from_vector(X[:, 0], title="Mixed Image 1")
    plt.savefig("mixed_src_1.png")
    show_image_from_vector(X[:, 1], title="Mixed image 2")
    plt.savefig("mixed_src_2.png")
    show_image_from_vector(X[:, 1], title="Mixed image 3")
    plt.savefig("mixed_src_3.png")
    # plt.show()

    # apply fastICA
    Xc = X - X.mean(axis=0)
    pca = PCA(whiten=True, random_state=0)
    Xw = pca.fit_transform(Xc)
    ica = FastICA(n_components=2, whiten=False, random_state=0)
    S_est = ica.fit_transform(Xw)
    A_est = ica.mixing_

    show_image_from_vector(S_est[:, 0], title="Recovered Image 1")
    show_image_from_vector(S_est[:, 1], title="Recovered Image 2")
    show_image_from_vector(S_est[:, 2], title="Recovered Image 23")
    # plt.show()