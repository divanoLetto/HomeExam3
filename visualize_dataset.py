from PIL import Image
import numpy as np

from deshuffler import deshuffler
from load_data import load_data
from shuffler import shuffler


def visualize(train_images, labels, n=10):
    '''
        Shows one by one the images in the dataset and print their label.
    '''
    num_images = train_images.shape[0]
    iterations = min(num_images, n)
    for i in range(iterations):
        image = train_images[i]
        label = labels[i]
        t_image = image.transpose((1, 2, 0))
        pil_image = Image.fromarray(np.uint8(t_image))
        print("Label for this image is: ", label)
        pil_image.show()
        input()


def visualize_permutations(train_images, labels, n=10):
    '''
        Shows one by one the images in the dataset, their label, their deshuffled version and their shuffled back version.
    '''
    num_images = train_images.shape[0]
    iterations = min(num_images, n)
    for i in range(iterations):
        image = train_images[i]
        t_image = image.transpose((1,2,0))
        pil_image = Image.fromarray(np.uint8(t_image))

        label = labels[i]
        print("Label for this image is: ", label)
        reorder_image = deshuffler(image, label)
        t_reorder_image = reorder_image.transpose((1, 2, 0))
        pil_reorder_image = Image.fromarray(np.uint8(t_reorder_image))

        disordered_image_again = shuffler(reorder_image, label)
        t_disordered_image_again = disordered_image_again.transpose((1, 2, 0))
        pil_disordered_image_again = Image.fromarray(np.uint8(t_disordered_image_again))

        Image.fromarray(np.hstack((np.array(pil_image), np.array(pil_reorder_image), np.array(pil_disordered_image_again)))).show()
        input()


if __name__ == "__main__":
    # Load Dataset 1
    train_images, train_labels, _, _ = load_data("DataShuffled.npz")
    # Visualize dataset
    visualize_permutations(train_images, train_labels, n=10)
    # Load Dataset 2
    train_images, train_labels, _, _ = load_data("DataNormal.npz")
    # Visualize dataset
    visualize(train_images, train_labels, n=10)



