import os
import torch  # PyTorch package
import torchvision.transforms as transforms  # transform data
import numpy as np  # for transformation
from PIL import Image
from src.cnn import Net
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def imshow(img):
    imgplot = plt.imshow(img)
    plt.show()

classes = (
    "airplanes",
    "cars",
    "birds",
    "cats",
    "deer",
    "dogs",
    "frogs",
    "horses",
    "ships",
    "trucks",
)

def test_single_file(n: Net):
    path = "data/test/dogs.jpeg"

    tf = transforms.Compose(
        [
            transforms.Resize((32,32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    true_class = 'dogs'
    img = Image.open(path)
    img_tf = tf(img).float().unsqueeze(0)
    imshow(img)
    with torch.no_grad():
        outputs = n(img_tf)
        _, y_pred = torch.max(outputs.data, 1)
        predicted_class = classes[y_pred[0]]
        print("This image is of", predicted_class)
        assert(predicted_class == true_class)

if __name__ == "__main__":
    if os.path.isfile("cifar_net.pth"):
        print("Model has been trained already")
    else:
        print("Train a model first")
     
    PATH = './cifar_net.pth'
    n = Net()
    n.load_state_dict(torch.load(PATH))

    test_single_file(n)

    print("Everything passed")

