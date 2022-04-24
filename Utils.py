def rescale(img_tensor):
    """
        rescale the tensor from [0,255] to [0,1]
    """
    ten = img_tensor/255
    return ten