"""C11 Assignment 3 Utility Functions"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import clip
from skimage.feature import hog
from PIL import Image
from typing import Optional
from scipy.ndimage import rotate
from sklearn.manifold import TSNE
import umap


#---------------------- CONFIG ---------------------------------

RANDOM_STATE = 42
HEIGHT = 64
WIDTH = 64
LABEL_MAP={0: 'muffin', 1: 'puppy'}
#---------------------- DATA LOADER ----------------------------
def load_dataset_muffin_puppy(
        data_folder: str,
        image_size: tuple[int, int] = (HEIGHT, WIDTH),
) -> tuple[np.ndarray, np.ndarray]:
    """ Return images stored in the directory data_folder and resize
        them to dimensions of image_size in a 2D array.
        
    Preconditions
    -------------
        data_folder structure must be
            data_folder/
            ├── muffin/
            └── puppy/
    
    Returns
    -------
    
    flattened images ready for ML models.
    """
    
    class_folders = LABEL_MAP
    
    all_images = []
    all_labels = []
    
    for label, class_name in class_folders.items():
        folder_path = os.path.join(os.getcwd(),data_folder, class_name)
        
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist!")
            continue
        
        print(f"Loading {class_name} images from {folder_path}...")
        
        image_count = 0
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(folder_path, filename)
                
                try:
                    # Load and convert to numpy array
                    img = Image.open(img_path)
                    img_array = np.array(img)
                    
                    # Verify shape 
                    expected_shape = (image_size)
                    if img_array.shape != expected_shape:
                        print(f"Warning: {filename} has shape" 
                              f"{img_array.shape}, expected {expected_shape}")
                        continue
                    
                    # Flatten to a vector
                    img_flat = img_array.flatten()
                    
                    all_images.append(img_flat)
                    all_labels.append(label)
                    image_count += 1
                    
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        print(f"  Loaded {image_count} {class_name} images")
    
    # Convert to numpy arrays
    X = np.array(all_images).astype(np.uint8)
    y = np.array(all_labels).astype(np.uint8)
    
    print("\nDataset summary:")
    print(f"  Total images: {len(X)}")
    print(f"  Shape: {X.shape}")
    print(f"  Labels: {np.bincount(y)} (muffins, puppies)")
    print(f"  Pixel range: [{X.min()}, {X.max()}]")
    
    return X, y

#------------- Feature Extraction ----------------------------
# extract HOG features
def extract_hog(
        images: np.ndarray, 
        height: int=HEIGHT, 
        width: int=WIDTH,
        orientation: int=9,
        pixels_per_cell: tuple[int, ...]=(8,8),
        cells_per_block: tuple[int, ...]=(2,2),
) -> np.ndarray:

    """Return HOG features for each flattened greyscale image in images.

    Parameters
    ----------
    images
        Shape (N, height * width), one flattened image per row.
    height, width
        Used to reshape each row back into a 2D image.
    orientation
        Number of gradient orientation bins.
    pixels_per_cell, cells_per_block
        Cell size in pixels and block size in cells, as (rows, cols).

    Returns
    -------
    np.ndarray
        Shape (N, num_hog_features), one feature row vecotr per image.

    """
    return np.array([
        hog(
            img.reshape(height, width),
            orientations=orientation,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
        )
        for img in images
    ])

# extract embeddings (i.e. features) from pre-trained deep learning model
def extract_clip(
        X_flat: np.ndarray, 
        img_shape: tuple = (HEIGHT, WIDTH),
        batch_size: int = 64,
) -> Optional[np.ndarray]:
    """Return normalized 512-dim CLIP embeddings for flattened greyscale images.

    Parameters
    ----------
    X_flat
        Shape (n_samples, height * width), dtype uint8.
    img_shape
        (height, width) used to unflatten each row.
    batch_size
        Number of images encoded per forward pass.

    Returns
    -------
    np.ndarray or None
        Float32 array of shape (n_samples, 512), or None if the CLIP
        dependencies are missing or extraction fails.
    """

    try:
        import torch
        import clip
        from PIL import Image as PILImage

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading CLIP model (this may take a moment)…")
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()

        height, width = img_shape
        embeddings = []

        with torch.no_grad():
            for i in range(0, len(X_flat), batch_size):
                batch = X_flat[i:i + batch_size]
                pil_imgs = [
                    PILImage.fromarray(
                        img.reshape(height, width).astype(np.uint8)
                    ).convert("RGB")
                    for img in batch
                ]
                imgs_tensor = torch.stack(
                    [preprocess(img) for img in pil_imgs]
                ).to(device)
                feats = model.encode_image(imgs_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)  # normalize
                embeddings.append(feats.cpu().numpy())

        return np.concatenate(embeddings, axis=0).astype(np.float32)

    except ImportError:
        print("  [CLIP] torch/clip not installed — "
              "skipping CLIP feature extraction.")
        return None
    except Exception as e:
        print(f"  [CLIP] Error: {e} — skipping CLIP feature extraction.")
        return None

#-------------- Noise Generators -------------------------------------
def flip_labels(
        y: np.ndarray, 
        flip_ratio: float = 0.2, 
        random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Return a copy of binary labels y with a fraction of them flipped.

    Precondition
    ------------
        y is 1D and contains only 0 and 1, and
        0 <= flip_ratio <= 1

    Parameters
    ----------
    y
        1D array of binary labels (0 or 1).
    flip_ratio
        Fraction of labels to flip.
    random_state
        Seed for the random number generator.

    Returns
    -------
    np.ndarray
        1D array of the same length as y, with int(len(y) * flip_ratio)
        randomly chosen labels flipped.
    """
    
    rng = np.random.default_rng(random_state)
    y_noisy = y.copy()
    n_flip = int(len(y) * flip_ratio)
    flip_indices = rng.choice(len(y), n_flip, replace=False)
    y_noisy[flip_indices] = 1 - y_noisy[flip_indices]
    return y_noisy

def reduce_contrast(X: np.ndarray, factor: float) -> np.ndarray:
    """Return X with contrast reduced by scaling pixels toward their mean.

    Precondition
    ------------
        X is 2D with dtype uint8, and 
        0 <= factor <= 1

    Parameters
    ----------
    X
        Shape (n_samples, n_features), one flattened image per row.
    factor
        Contrast reduction amount. 0 keeps the original contrast,
        1 gives no contrast (fully gray).

    Returns
    -------
    np.ndarray
        Same shape as X, dtype uint8, values in [0, 255].
    """
    mean = X.mean(axis=1, keepdims=True)
    X_reduced = mean + (1 - factor) * (X.astype(np.float32) - mean)
    return np.clip(X_reduced, 0, 255).astype(np.uint8)




# rotating an image angle degrees
def add_rotation(
        X_flat: np.ndarray,
        img_shape: tuple[int, int] = (HEIGHT, WIDTH),
        angle: float = 30,
) -> np.ndarray:
    """Return a copy of X_flat with each image rotated by angle degrees.

    Precondition
    ------------
        X_flat is 2D with dtype uint8, and
        X_flat.shape[1] == img_shape[0] * img_shape[1]

    Parameters
    ----------
    X_flat
        Shape (n_samples, height * width), one flattened image per row.
    img_shape
        (height, width) used to unflatten each row.
    angle
        Rotation angle in degrees.

    Returns
    -------
    np.ndarray
        Same shape as X_flat, dtype uint8.
        
    Notes
    -----
        The image size is kept the same (reshape=False), so corners that
        rotate out of frame are cropped and the empty areas are filled with 0.
    """
    rotated = []
    for img in X_flat:
        img_2d = img.reshape(img_shape)
        img_rot = rotate(img_2d, angle, reshape=False)
        rotated.append(img_rot.flatten())
    return np.array(rotated)

#-------------- Text Output Helper -----------------------------    
def compute_concept_scores(
        clip_features: np.ndarray,
        concepts: list[str],
        clip_model_name: str = "ViT-B/32",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> np.ndarray:
    """
    Project CLIP image features onto a set of text concept
    directions to produce interpretable concept scores.

    Parameters
    ----------
    clip_features  
        Shape (n_samples, n_dims)
        CLIP image feature matrix, as returned by
        clip.model.encode_image. Each row should already be unit-normalized.
    concepts 
        List of text concepts to project onto (e.g. "a dog face", "fur texture").
        Each string is tokenized and encoded using the CLIP text encoder.
    clip_model_name
        Name of the CLIP model variant to load. Default is "ViT-B/32".
        Other options include "ViT-L/14", "RN50", "RN101".
    device
        Device to run the CLIP text encoder on. Defaults to "cuda" if
        available, otherwise "cpu".

    Returns
    -------
    concept_scores
        Shape (n_samples, n_concepts)
        Cosine similarity scores between each image and each text concept.
        Values are in [-1, 1] since both image and text features are
        L2-normalized. Higher values indicate stronger alignment with
        the concept.
    Notes
    -----
    Handles both L2-normalized and non-nomalized clip features
    """
    # Load CLIP model
    model, _ = clip.load(clip_model_name, device=device)
    model.eval()

    # Encode and normalize text concepts
    text_tokens = clip.tokenize(concepts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens) # (n_concepts, n_dims)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # normalize the clip features
    clip_features = clip_features / np.linalg.norm(clip_features, axis=1, keepdims=True)
    
    # Project image features onto concept directions
    text_features_np = text_features.cpu().numpy() # (n_concepts, n_dims)
    concept_scores   = clip_features @ text_features_np.T # (n_samples, n_concepts)

    return concept_scores

#-------------- Visualization  ----------------------------------------
def show_image(
        x: np.ndarray, 
        y: int, 
        shape: tuple[int, ...] = (HEIGHT, WIDTH), 
        cmap: str = 'gray',
) -> None:
    """Display one flattened image with its label as the plot title.

    Precondition
    ------------
        x.size == product of shape, and y is a key of LABEL_MAP

    Parameters
    ----------
    x
        One flattened image, uint8 pixel values.
    y
        Class label of the image, used to look up its name in LABEL_MAP.
    shape
        Shape to reshape x into, such as (height, width).
    cmap
        Matplotlib colormap name.
    """
    x_reshaped = x.reshape(shape)
    plt.imshow(x_reshaped.astype(np.uint8), cmap=cmap)
    plt.title(f"Label is {y}: {LABEL_MAP[y]}")
    plt.show()
    
def visualize_hog_sample(
    pixel_vector: np.ndarray,
    shape: tuple[int, int] = (HEIGHT, WIDTH),
    title: str = "Image and HOG",
    orientation: int = 9,
    pixels_per_cell: tuple[int, int] = (4, 4),
    cells_per_block: tuple[int, int] = (2, 2),
) -> np.ndarray:
    """Plot an image next to its HOG sketch and return its HOG features.

    Precondition
    ------------
        pixel_vector.size == shape[0] * shape[1]

    Parameters
    ----------
    pixel_vector
        One flattened greyscale image.
    shape
        (height, width) used to unflatten pixel_vector.
    title
        Title shown above both plots.
    orientation
        Number of gradient orientation bins.
    pixels_per_cell, cells_per_block
        Cell size in pixels and block size in cells, as (rows, cols).

    Returns
    -------
    np.ndarray
        1D array of HOG features for the image.
    """
    image_2d = pixel_vector.reshape(shape)

    features, hog_image = hog(
        image_2d,
        orientations=orientation,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=True,
    )

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8, 4), sharex=True, sharey=True
    )

    ax1.axis("off")
    ax1.imshow(image_2d, cmap="gray")
    ax1.set_title("Original Image")

    ax2.axis("off")
    ax2.imshow(hog_image, cmap="gray")
    ax2.set_title("HOG Structure (Sketch)")

    plt.suptitle(title)
    plt.show()

    return features

def visualize_embeddings(
    embeddings: np.ndarray,
    y: np.ndarray,
    feature_name: str = "UNKNOWN",
    labels: dict[int, str] | None = LABEL_MAP,
    method: str = "tsne",
) -> None:
    """Plot embeddings in 2D, colored by class.

    Precondition
    ------------
        embeddings is 2D and len(y) == len(embeddings)

    The embeddings are reduced to two components with t-SNE or UMAP
    (random_state=RANDOM_STATE for reproducibility), 
    and one scatter series is drawn per class.

    Parameters
    ----------
    embeddings
        Shape (n_samples, n_features).
    y
        Shape (n_samples,), the class label of each sample.
    feature_name
        Name of the feature type, used in the plot title.
    labels
        Maps class index to display name, such as {0: "muffin", 1: "puppy"}.
        A class missing from labels (or labels being None) is shown by its
        index.
    method
        Reduction method, "tsne" or "umap".

    Raises
    ------
    ValueError
        If method is not "tsne" or "umap".
    """
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=RANDOM_STATE)
    elif method == "umap":
        reducer = umap.UMAP(random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown method: {method}, choose 'tsne' or 'umap'")

    embeddings_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    for class_idx in np.unique(y):
        mask = y == class_idx
        name = (
            labels[class_idx]
            if labels and class_idx in labels
            else str(class_idx)
        )
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=name,
            alpha=0.6,
        )

    plt.legend()
    plt.title(f"{feature_name} Features ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.show()


# %%
