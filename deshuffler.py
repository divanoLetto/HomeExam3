import numpy as np


def deshuffler(img_s, order):
    """ 
    Takes a shuffled image and reshuffles it according to a given order.
    img_s: A shuffled image of size (3x96x96).
    order: Order of patches from shuffled image in dehuffled image of size (4, ).
    """
    
    img_ns = np.zeros_like(img_s)

    num_tiles = 2
    tile_size = 96 // num_tiles

    starting_points = []
    for i in range(num_tiles):
        for j in range(num_tiles):
            starting_points.append((i*tile_size, j*tile_size))

    for i in range(num_tiles**2):

        x_idx = starting_points[i][0]
        y_idx = starting_points[i][1]

        x_idx_p = starting_points[order[i]][0]
        y_idx_p = starting_points[order[i]][1]

        img_ns[:, x_idx_p:x_idx_p+tile_size, y_idx_p:y_idx_p+tile_size] = img_s[:, x_idx:x_idx+tile_size, y_idx:y_idx+tile_size]


    return img_ns
