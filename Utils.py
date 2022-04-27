import torch
import matplotlib.pyplot as plt


def rescale(img_tensor):
    """
        rescale the tensor from [0,255] to [0,1]
    """
    ten = img_tensor/255
    return ten


def toTensor(img_tensor):
    """
        return a PyTorch tensor
    """
    ten = torch.Tensor(img_tensor)
    return ten


def my_plot(train_y, valid_y, num_epochs, label):
    plt.xlabel('Epoch')
    plt.ylabel(label)
    x = list(range(0, num_epochs))
    plt.plot(x, train_y, label='train')
    plt.plot(x, valid_y, label='validation')
    plt.legend()
    plt.show()


def quarter_accuracy(true_perm, pred_perm):
    count = 0
    for i in range(len(true_perm)):
        l_count = 0
        t_p = [int(k) for k in true_perm[i] if k.isdigit()]
        p_p = [int(k) for k in pred_perm[i] if k.isdigit()]
        for j in range(len(t_p)):
            if t_p[j] == p_p[j]:
                l_count += 1
        count += l_count/len(t_p)
    count = count/ len(true_perm)
    return count



