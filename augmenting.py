from itertools import permutations
import random
import numpy as np
from deshuffler import deshuffler
from shuffler import shuffler


def analize_dataset(labels):
    '''
    Prints how many sample for class are present in the dataset labels.
    '''
    print("Samples for each class: ")
    class_count = {}
    for lab in list(labels):
        s_lab = str(lab)
        if s_lab not in class_count.keys():
            class_count[s_lab] = 0
        class_count[s_lab] += 1

    for key, value in class_count.items():
        print("Class ", key," : ",value)


def basic_augmentation(train_images, labels):
    '''
    :param train_images: trainset of images
    :param labels: labels of the trainset
    :return: (new_train_images, new_train_labels) augmented dataset generating for each image also the deshuffled version of it.
    '''
    ordered_set = np.array([0, 1, 2, 3])
    new_train_images = []
    new_train_labels = []
    num_images = train_images.shape[0]
    for i in range(num_images):
        image = train_images[i]
        label = labels[i]
        if not np.equal(ordered_set, label).all():
            reorder_image = deshuffler(image, label)
            reorder_label = np.array([0,1,2,3])

            new_train_images.append(reorder_image)
            new_train_labels.append(reorder_label)

        new_train_images.append(image)
        new_train_labels.append(label)

    return np.array(new_train_images), np.array(new_train_labels)


def super_augmentation(train_images, labels):
    '''
    :param train_images: trainset of images
    :param labels: labels of the trainset
    :return: (new_train_images, new_train_labels) augmented dataset generating for each image all his 24 shuffle permutation images.
    '''
    new_train_images = []
    new_train_labels = []
    num_images = train_images.shape[0]
    for i in range(num_images):
        image = train_images[i]
        label = labels[i]

        reorder_image = deshuffler(image, label)
        perms = list(permutations(range(0, 4)))

        for perm in perms:
            disordered_image = shuffler(reorder_image, perm)
            new_train_images.append(disordered_image)
            new_train_labels.append(perm)

    return np.array(new_train_images), np.array(new_train_labels)


def shuffle_dataset(train_images, labels):
    c = list(zip(train_images.tolist(), labels.tolist()))
    random.shuffle(c)
    new_train_images, new_train_labels = zip(*c)
    return np.array(new_train_images), np.array(new_train_labels)