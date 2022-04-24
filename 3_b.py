from torchvision.models import vgg11_bn, vgg16
from Utils import rescale
from load_data import load_data
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelBinarizer
from torchvision.transforms import transforms
import numpy as np
from sklearn.metrics import accuracy_score


def train_session(model, dataloader, args):
    num_epochs = args['num_epochs']
    save_freq = args['save_freq']
    weights_save_path = args['weights_save_path']
    print_freq = args['print_freq']

    model.train()  # Set model to training mode
    softmax = torch.nn.Softmax(dim=1)

    # todo SGD o Adam è l'opzione migliore? learning rate?
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # optim.Adam(model.parameters(), lr=0.05)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        running_loss = 0.0

        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            outputs = softmax(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % print_freq == print_freq - 1:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_freq:.5f}')
                running_loss = 0.0

        if epoch % save_freq == save_freq - 1:
            print("Model saved at epoch: ", epoch)
            torch.save(model.state_dict(), weights_save_path)

    print('Finished Training')
    torch.save(model.state_dict(), weights_save_path)


def evaluate(model, dataloader, args):
    losses = []
    top1 = []

    # Set to evaluate mode
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data[0], data[1]

            # compute output
            output = model(images)
            output_s = softmax(output)
            # measure loss
            loss = criterion(output_s, labels)
            # measure accuracy
            acc = accuracy_score(np.argmax(labels, axis=1), np.argmax(output_s, axis=1))

            losses.append(loss)
            top1.append(acc)

            if i % args['print_freq'] == 0:
                print("Accuracy: ", acc, " ,loss: ", loss)

    return np.average(losses), np.average(top1)


if __name__ == "__main__":
    # settings
    num_classes = 24  # 24 = 4!
    num_epochs = 10000
    batch_size = 128
    print_freq = 5
    save_freq = 100
    train = True
    weights_save_path = 'model/weights_5000ep_VGG11.pth'

    # Load Dataset
    train_images, train_labels, test_images, test_labels = load_data("DataShuffled.npz")
    # transform labels data considering them as categorical strings
    train_labels_coded = [str(s) for s in train_labels.tolist()]
    lb = LabelBinarizer()
    lb.fit(train_labels_coded)

    # todo questi valori vanno bene anche per il dataset?
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transff = transforms.Compose([
        transforms.Lambda(rescale),
        normalize,
    ])

    # Crate a Model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = vgg16(pretrained=False)  # todo probabilmente farla pretrained è meglio
    model.classifier[6] = torch.nn.Linear(4096, num_classes)
    print(model)

    if train:
        model.to(device)
        onehot_trainset = lb.transform(train_labels_coded)
        #  per reverse lb.inverse_transform(encoded_labels)
        # transform to torch tensor
        train_img_ten = torch.Tensor(train_images)
        train_label_ten = torch.Tensor(onehot_trainset)
        # process and normalize the images
        train_img_ten_proc = transff(train_img_ten)
        dataset = TensorDataset(train_img_ten_proc, train_label_ten)  # create your datset
        train_dataloader = DataLoader(dataset, batch_size=batch_size)

        args = {}
        args['num_epochs'] = num_epochs
        args['save_freq'] = save_freq
        args['weights_save_path'] = weights_save_path
        args['print_freq'] = print_freq

        train_session(
            model=model,
            dataloader=train_dataloader,
            args=args
        )
    else:
        model.load_state_dict(torch.load(weights_save_path))

    # Evaluate
    test_labels_coded = [str(s) for s in test_labels.tolist()]
    lb.fit(test_labels_coded)
    onehot_testset = lb.transform(test_labels_coded)
    # transform to torch tensor
    test_img_ten = torch.Tensor(test_images)
    test_label_ten = torch.Tensor(onehot_testset)
    # process and normalize the images
    test_img_ten_proc = transff(test_img_ten)

    test_set = TensorDataset(test_img_ten, test_label_ten)
    test_dataloader = DataLoader(test_set, batch_size=batch_size)

    args = {}
    args['print_freq'] = print_freq
    losses_avg, top1_avg = evaluate(
        model=model,
        dataloader=test_dataloader,
        args=args
    )
    print("Average loss: ", losses_avg)
    print("Average top1 acc: ", top1_avg)
