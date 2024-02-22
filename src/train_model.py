from cnn import Net
import torch  # PyTorch package
import torch.nn as nn  # basic building block for neural neteorks
import torch.nn.functional as F  # import convolution functions like Relu
import torch.optim as optim  # optimzer

def train(classes, train_loader):
    dataiter = iter(train_loader)
    _, labels = next(dataiter)
    print(" ".join("%s" % classes[labels[j]] for j in range(4)))

    n = Net()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(n.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(n.parameters(), lr=0.001)
    loss_tracker = {}
    epochs = 20

    for epoch in range(epochs):
        running_loss = 0.0
        running_loss_tracker = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = n(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 49:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 50))
                running_loss_tracker += running_loss / 50
                running_loss = 0.0

        running_loss_tracker = running_loss_tracker / 5
        print("Epoch %d loss: %.3f" % (epoch + 1, running_loss_tracker))
        loss_tracker[epoch + 1] = running_loss_tracker

    print("Finished Training")
    print(loss_tracker)


    PATH = './cifar_net.pth'
    torch.save(n.state_dict(), PATH)
