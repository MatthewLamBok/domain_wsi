# Import the recommended normalization technique for stardist.
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from skimage.measure import regionprops, label
from skimage.morphology import closing, square
from scipy.spatial.distance import cdist
import numpy as np

import matplotlib.pyplot as plt

# Import squidpy and additional packages needed for this tutorial.
import squidpy as sq


img = sq.datasets.visium_hne_image_crop()
print(type(img))
crop = img.crop_corner(0, 0, size=1000)
crop.show("image")


def stardist_2D_versatile_he(img, nms_thresh=None, prob_thresh=None):
    # axis_norm = (0,1)   # normalize channels independently
    axis_norm = (0, 1, 2)  # normalize channels jointly
    # Make sure to normalize the input image beforehand or supply a normalizer to the prediction function.
    # this is the default normalizer noted in StarDist examples.
    img = normalize(img, 1, 99.8, axis=axis_norm)
    model = StarDist2D.from_pretrained("2D_versatile_he")
    labels, _ = model.predict_instances(
        img, nms_thresh=nms_thresh, prob_thresh=prob_thresh
    )

    # Process labels to extract features
    props = regionprops(labels)
    
    cell_sizes = [prop.area for prop in props]  # Size of each cell
    num_cells = len(props)  # Number of cells
    centroids = np.array([prop.centroid for prop in props])  # Centroids of each cell
    
    # Calculate pairwise distances between cell centroids
    distances = cdist(centroids, centroids)
    
    # Additional information, such as perimeter, circularity, etc.
    cell_perimeters = [prop.perimeter for prop in props]
    cell_circularity = [4 * np.pi * prop.area / (prop.perimeter ** 2) if prop.perimeter > 0 else 0 for prop in props]
    
    # Compile information into a dictionary for easy access
    cell_info = {
        'cell_sizes': cell_sizes,
        'num_cells': num_cells,
        'cell_perimeters': cell_perimeters,
        'cell_circularity': cell_circularity,
        'distances': distances,
    }
    print(len(cell_info['cell_sizes']))
    #print(len(cell_info['num_cells']))
    print(len(cell_info['cell_perimeters']))
    print(len(cell_info['cell_circularity']))
    print(len(cell_info['distances']))

    return labels, cell_info

sq.im.segment(
    img=crop,
    layer="image",
    channel=None,
    method=stardist_2D_versatile_he,
    layer_added="segmented_stardist",
    prob_thresh=0.3,
    nms_thresh=None,
)

print(crop)
print(f"Number of segments in crop: {len(np.unique(crop['segmented_stardist']))}")

fig, axes = plt.subplots(1, 2)
crop.show("image", ax=axes[0])
_ = axes[0].set_title("H&H")
crop.show("segmented_stardist", cmap="jet", interpolation="none", ax=axes[1])
_ = axes[1].set_title("segmentation")
#plt.show()
