import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random


from hw1_314734724_train_q1 import NN_function, NN


def evaluate_hw1():
    # load networks parameters
    params = torch.load('q2_weight.pkl')

    # load MNIST data set
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)])
    test_data = dsets.MNIST(root='./data', train=False, download=True, transform=transform)
    X, _ = NN_function.data_to_torch(test_data)
    test_size = len(X)
    Y = torch.FloatTensor([random.randint(0, 1) for _ in range(test_size)])
    results = NN.predict_and_evaluate(X, Y, params)

    print(f"test accuaracy: {results[0]}")
    print(f"test CE loss: {results[1]}")

evaluate_hw1()