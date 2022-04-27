from sklearn.neighbors import KNeighborsClassifier
from torchvision import transforms
from torchvision.models import vgg11_bn
from Utils import toTensor, rescale
from load_data import load_data
import torch
from sklearn.metrics import accuracy_score
import numpy as np


def NeighborsClassifier_from_model(model, p_train_images, train_labels, p_test_images, test_labels):
    # is computational heavy pass all the images in one step to the model, lets to batches
    model_train_results = []
    for i in range(0, len(p_train_images), batch_size):
        images = p_train_images[i:i + batch_size]
        # Get the feature vector from the model
        train_output = model(images)
        model_train_results.append(train_output.detach().numpy())
    # stack all the results obtained
    train_outputs = np.vstack((n for n in model_train_results))

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_outputs, train_labels)

    model_test_results = []
    for i in range(0, len(p_test_images), batch_size):
        images = p_test_images[i:i + batch_size]
        # Get the feature vector from the model
        test_output = model(images)
        model_test_results.append(test_output.detach().numpy())

    test_outputs = np.vstack((n for n in model_test_results))
    # predict the class from the feature vector
    predictions = neigh.predict(test_outputs)

    acc = accuracy_score(test_labels, predictions)
    return acc


if __name__ == "__main__":
    # settings
    num_classes = 24  # 24 = 4!
    batch_size = 32

    weights_load_path = 'model/weights.pth'

    # Load Dataset
    train_images, train_labels, test_images, test_labels = load_data("DataNormal.npz")
    num_train_samples = train_images.shape[0]
    num_test_samples = test_images.shape[0]

    # Crate a trained Model
    model = vgg11_bn()
    # Change the last layer to match the dimentions of the saved weights
    model.classifier[6] = torch.nn.Linear(4096, 24)
    # Load the weights based on the other Dataset
    model.load_state_dict(torch.load(weights_load_path))
    # Remove the last classification FC layer from the model. The output now is a (num_images,4096) feature vector
    model.classifier = torch.nn.Sequential(*[model.classifier[i] for i in range(4)])
    print("Model trained on the DataShuffled Dataset: ")
    print(model)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transff = transforms.Compose([
        transforms.Lambda(toTensor),
        transforms.Lambda(rescale),
        normalize,
    ])
    # process the images
    processed_train_images = transff(train_images)
    processed_test_images = transff(test_images)
    # Get the accuracy from this model on the DataNormal Dataset
    acc1 = NeighborsClassifier_from_model(model=model,
                                          p_train_images=processed_train_images,
                                          train_labels=train_labels,
                                          p_test_images=processed_test_images,
                                          test_labels=test_labels)
    print("Accuracy with feature extraction from the trained model: ", acc1)

    # Crate a Model with random weights
    model = vgg11_bn()
    model.classifier[6] = torch.nn.Linear(4096, 24)
    model.classifier = torch.nn.Sequential(*[model.classifier[i] for i in range(4)])
    acc2 = NeighborsClassifier_from_model(model=model,
                                          p_train_images=processed_train_images,
                                          train_labels=train_labels,
                                          p_test_images=processed_test_images,
                                          test_labels=test_labels)
    print("Accuracy with feature extraction from the random weights model: ", acc2)

    neigh = KNeighborsClassifier(n_neighbors=3)

    flatten_image_dim = np.prod(train_images.shape[1:])
    # Flatten the images to be able to use them directly as a feature vector
    flatten_train_images = train_images.reshape((num_train_samples, flatten_image_dim))
    flatten_test_images = test_images.reshape((num_test_samples, flatten_image_dim))

    neigh.fit(flatten_train_images, train_labels)
    predictions = neigh.predict(flatten_test_images)

    acc3 = accuracy_score(test_labels, predictions)
    print("Accuracy with nearest neighbor classifier trained directly on the input space: ", acc3)




