import torch
import torchvision
import torchvision.transforms as transforms

if __name__ == "__main__":
    """
    A convolutional classifier will need:
    Input
        torch.Size([3, 32, 32])
        torch.float32
    Output
        tensor(5) # class no.6
        torch.int64
    """
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(
        root="./src/images/data", transform=transform, train=True
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)
    valset = torchvision.datasets.CIFAR10(
        root="./src/images/data", train=False, transform=transform
    )
    valloader = torch.utils.data.DataLoader(valset, batch_size=8, shuffle=True)
    
    ##############
    
    
    
    ###########
    for x, y in dataloader:
        print(x[0].shape)
        print(x[0].dtype)
        print(y[0].shape)
        print(y[0].dtype)
        break
