# python
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torchvision.datasets
from matplotlib import pyplot as plt

device = "mps" if torch.mps.is_available() else "cpu"

n_epochs = 5
batch_size_train = 128
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "/tmp/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size_train,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "/tmp/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size_test,
    shuffle=True,
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor):
        x = nnf.relu(nnf.max_pool2d(self.conv1(x), 2))
        x = nnf.relu(nnf.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = nnf.relu(self.fc1(x))
        x = nnf.dropout(x, training=self.training)
        x = self.fc2(x)
        return nnf.log_softmax(x, dim=1)


if __name__ == "__main__":
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])

    plt.show()

    network = Net()
    network.to(device)
    optimizer = torch.optim.SGD(
        network.parameters(), lr=learning_rate, momentum=momentum
    )

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


    def train(epoch: int):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = network(data)
            loss = nnf.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * batch_size_train)
                    + ((epoch - 1) * len(train_loader.dataset))
                )
                torch.save(network.state_dict(), "./results/model.pth")
                torch.save(optimizer.state_dict(), "./results/optimizer.pth")


    def test():
        network.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = network(data)
                test_loss += nnf.nll_loss(output, target, reduction="sum").item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print(
            "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                int(correct),
                len(test_loader.dataset),
                100.0 * float(correct) / len(test_loader.dataset),
            )
        )


    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color="blue")
    plt.scatter(test_counter, test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("number of training examples seen")
    plt.ylabel("negative log likelihood loss")

    plt.show()

    with torch.no_grad():
        example_data_gpu = example_data.to(device)
        output = network(example_data_gpu)

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
        plt.title(
            "Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item())
        )
        plt.xticks([])
        plt.yticks([])

    plt.show()
