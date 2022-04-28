import argparse
from torchvision.models import vgg11_bn, vgg16
from Utils import rescale, toTensor, my_plot, quarter_accuracy
from augmenting import basic_augmentation, analize_dataset, super_augmentation, shuffle_dataset
from load_data import load_data
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import LabelBinarizer
from torchvision.transforms import transforms
import numpy as np
from sklearn.metrics import accuracy_score
from utils import settings_parser


def get_dataset(images, labels, lb, transformation):
    # transform labels data considering them as categorical strings
    labels_coded = [str(s) for s in labels.tolist()]

    labels_onehot = lb.transform(labels_coded)
    processed_images = transformation(images)

    dataset = TensorDataset(processed_images, torch.Tensor(labels_onehot))
    return dataset


def train_session(model, train_dataloader, args, val_dataloader=None):
    num_epochs = args['num_epochs']
    weights_save_path = args['weights_save_path']
    print_freq = args['print_freq']
    len_trainset = args['len_trainset']
    len_validationset = args['len_validationset']

    min_valid_loss = np.inf

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.005)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        tmp_loss = 0.0
        running_loss = 0.0

        model.train()
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # save statistics
            batch_loss = loss.item()
            tmp_loss += batch_loss
            running_loss += batch_loss

            if i % print_freq == print_freq - 1:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {tmp_loss / print_freq:.5f}')
                tmp_loss = 0

        train_losses.append(running_loss / len(train_dataloader))

        if val_dataloader is not None:
            valid_loss = 0

            model.eval()
            for i, data in enumerate(val_dataloader, 0):

                images, labels = data[0].to(device), data[1].to(device)
                target = model(images)
                val_loss = criterion(target, labels)
                # save statistics
                valid_loss += val_loss.item()

            validation_losses.append(valid_loss / len(val_dataloader))

            print(f'Epoch {epoch + 1} \t\t Training Loss: {running_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss / len(val_dataloader)}')
            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss
                # Saving State Dict
                torch.save(model.state_dict(), weights_save_path)

    print('Finished Training')
    torch.save(model.state_dict(), weights_save_path)
    return train_losses, validation_losses


def evaluate(model, dataloader, lb, args):
    losses = []
    top1 = []
    quarter_acc = []

    # Set to evaluate mode
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data[0], data[1]
            # compute output
            output = model(images)
            # measure loss
            loss = criterion(output, labels)
            # measure accuracy
            acc = accuracy_score(np.argmax(labels, axis=1), np.argmax(output, axis=1))

            true_perm = lb.inverse_transform(labels.numpy())
            max_idx_output = output.argmax(1)
            labels_onehot = torch.zeros(output.shape).scatter(1, max_idx_output.unsqueeze(1), 1.0)
            pred_perm = lb.inverse_transform(labels_onehot.numpy())
            quart_acc = quarter_accuracy(true_perm.tolist(), pred_perm.tolist())

            losses.append(loss.item())
            top1.append(acc)
            quarter_acc.append(quart_acc)

            if i % args['print_freq'] == 0:
                print("Accuracy: ", acc, " loss: ", loss)

    return np.average(losses), np.average(top1), np.average(quart_acc)


if __name__ == "__main__":
    # settings
    parser = argparse.ArgumentParser()
    # Get settings
    settings_system = settings_parser.get_settings('System')
    settings_dataset = settings_parser.get_settings('Dataset')
    settings_model = settings_parser.get_settings('Model')

    print_freq = int(settings_system['print_freq'])
    save_freq = int(settings_system['save_freq'])

    split_train_val = float(settings_dataset['split_train_val'])
    weights_save_path = settings_dataset['weights_save_path']
    weights_load_path = settings_dataset['weights_load_path']
    dataset_path = settings_dataset['dataset_path']

    num_epochs = int(settings_model['num_epochs'])
    batch_size = int(settings_model['batch_size'])
    if settings_model['train'] == "True":
        train = True
    else:
        train = False

    num_classes = 24  # 24 = 4!

    # Load Dataset
    train_images, train_labels, test_images, test_labels = load_data(dataset_path)
    # One hot label encoder
    lb = LabelBinarizer()
    lb.fit([str(s) for s in test_labels.tolist()] + [str(s) for s in train_labels.tolist()])
    # define image processing
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transff = transforms.Compose([
        transforms.Lambda(toTensor),
        transforms.Lambda(rescale),
        normalize,
    ])
    # Create a Model
    model = vgg11_bn(pretrained=True)
    model.classifier[6] = torch.nn.Linear(4096, num_classes)
    print("Model")
    print(model)

    if train:
        print("Train: ")
        # Set GPU if possible
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        # split in train and validation test
        dim_dataset = train_images.shape[0]
        dim_train = int(dim_dataset * split_train_val)
        dim_val = dim_dataset - dim_train

        valid_images = train_images[dim_train:]
        valid_labels = train_labels[dim_train:]
        train_images = train_images[0:dim_train]
        train_labels = train_labels[0:dim_train]
        # Augment dataset
        train_images, train_labels = super_augmentation(train_images, train_labels)

        print("Train set dim: ", train_images.shape[0], " test set dim: ", valid_images.shape[0])
        # Get train and validation datasets
        train_dataset = get_dataset(images=train_images, labels=train_labels, lb=lb, transformation=transff)
        val_dataset = get_dataset(images=valid_images, labels=valid_labels, lb=lb, transformation=transff)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        args = {}
        args['num_epochs'] = num_epochs
        args['save_freq'] = save_freq
        args['weights_save_path'] = weights_save_path
        args['print_freq'] = print_freq
        args['len_trainset'] = dim_train
        args['len_validationset'] = dim_val

        # train model
        train_losses, valid_losses = train_session(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            args=args
        )
        my_plot(train_y=train_losses, valid_y=valid_losses, num_epochs=num_epochs, label="Loss")

    # Evaluate

    print("Evaluate: ")
    model.to('cpu')
    model.load_state_dict(torch.load(weights_load_path))
    # onehot encoding for test labels
    test_labels_coded = [str(s) for s in test_labels.tolist()]
    test_labels_onehot = lb.transform(test_labels_coded)
    # transform to torch tensor
    test_label_ten = torch.Tensor(test_labels_onehot)
    # process and normalize the images
    test_img_processed = transff(test_images)

    test_set = TensorDataset(test_img_processed, test_label_ten)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    args = {}
    args['print_freq'] = print_freq
    losses_avg, top1_avg, quart_acc_avg = evaluate(
        model=model,
        dataloader=test_dataloader,
        lb=lb,
        args=args
    )
    print("Average loss: ", losses_avg)
    print("Average top1 acc: ", top1_avg)
    print("Average quarter acc: ", quart_acc_avg)
