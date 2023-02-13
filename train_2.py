import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import numpy as np

from hw1_314734724_train_q1 import NN_function, NN

if __name__ == "__main__":
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)])
    train_data = dsets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = dsets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_size = len(train_data)
    test_size = len(test_data)

    # train data conversion
    X_train, _ = NN_function.data_to_torch(train_data)
    Y_train = torch.FloatTensor([random.randint(0, 1) for _ in range(train_size)])
    # test data conversion
    X_test, _ = NN_function.data_to_torch(test_data)
    Y_test = torch.FloatTensor([random.randint(0, 1) for _ in range(test_size)])

    e = 1000
    nn = NN(epochs=e, learning_rate=0.01, batch_size=128)

    acc_process_train, acc_process_test, loss_process_train, loss_process_test = nn.train_q2(X_train[0:128, :],
                                                                                             Y_train[0:128],
                                                                                             X_test, Y_test)
    lens = len(acc_process_train)
    # print results
    plt.plot(np.arange(lens) * 10, acc_process_train, label='train')
    plt.plot(np.arange(lens) * 10, acc_process_test, label='test')
    plt.xlabel('epocs')
    plt.ylabel('accuracy')
    plt.title('train and test accuracy')
    plt.legend()
    plt.show()

    plt.plot(np.arange(lens) * 10, loss_process_train, label='train')
    plt.plot(np.arange(lens) * 10, loss_process_test, label='test')
    plt.xlabel('epocs')
    plt.ylabel('cross entropy loss')
    plt.title('train and test loss')
    plt.legend()
    plt.show()

    # save model parameters
    with open('q2_weight.pkl', 'wb') as handle:
        pass
    torch.save(nn.params, 'q2_weight.pkl')

