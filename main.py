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

def load_and_prepare(path, size=(256, 256)):
    # load img
    img = Image.open(path).convert('L')

    # resize
    img = img.resize(size)

    # convert to numpy matrix
    arr = np.asarray(img, dtype=np.float32) / 255.0

    # flatten into 1D vector
    flat = arr.flatten()

    return arr, flat

img1_arr, img1_flat = load_and_prepare("src_6.png")
img2_arr, img2_flat = load_and_prepare("src_5.PNG")

print("Image 1 shape:", img1_arr.shape)
print("Flatten shape:", img1_flat.shape)

# stack into one mixed image S
# shape = (N, 2)
S = np.column_stack((img1_flat, img2_flat))
print("S shape: ", S.shape)

# Create mixing matrix
A = np.array([
    [1.0, 0.5],
    [0.4, 1.0]
])

# Generate the mixed signals: X = Sdet(A)
X = S @ A.T 
print("X shape:", X.shape)

show_image_from_vector(X[:,0], title="Mixed Image 1")
plt.savefig("mixed_src_1.png")
show_image_from_vector(X[:,1], title="Mixed image 2")
plt.savefig("mixed_src_2.png")
plt.show()

# apply fastICA
Xc = X - X.mean(axis=0)
pca = PCA(whiten=True, random_state=0)
Xw = pca.fit_transform(Xc)
ica = FastICA(n_components=2, whiten=False, random_state=0)
S_est = ica.fit_transform(Xw)
A_est = ica.mixing_

show_image_from_vector(S_est[:,0], title="Recovered Image 1")
show_image_from_vector(S_est[:,1], title="Recovered Image 2")
plt.show()