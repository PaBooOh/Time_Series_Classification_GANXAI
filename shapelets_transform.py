from sktime.transformations.panel.shapelet_transform import ShapeletTransform, RandomShapeletTransform
from config import *


# Random Shapelet Transform for saving time
def get_st(random=True, min_len=10, max_len=50):
    if random:
        st = RandomShapeletTransform(
        # n_shapelet_samples=20000,
        max_shapelets=20, # per class
        min_shapelet_length=min_len,
        max_shapelet_length=max_len,
        time_limit_in_minutes=1,
        )
    else:
        st = ShapeletTransform(
        max_shapelets_to_store_per_class=10,
        min_shapelet_length=10,
        max_shapelet_length=40,
        )
    return st


