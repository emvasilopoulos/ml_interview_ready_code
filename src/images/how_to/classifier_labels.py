import torch
import torchvision
import torchvision.transforms as transforms

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        root="./src/images/data", transform=transform
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    for x, y in dataloader:
        print(x[0].shape)
        print(x[0].dtype)
        print(y[0].shape)
        print(y[0].dtype)
        break
