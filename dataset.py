import torchvision
import torchvision.transforms as transforms
import torch


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# torchvision数据集的输出是在[0, 1]范围内的PILImage图片。
# 我们此处使用归一化的方法将其转化为Tensor，数据范围为[-1, 1]
def get_trainloader(batch_size = 4,transform=transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])):
    trainset = torchvision.datasets.CIFAR100(root='./CIFAR100', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                              shuffle=True, num_workers=4)
    return trainloader
def get_testloader(batch_size = 4,transform=transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])):
    testset = torchvision.datasets.CIFAR100(root='./CIFAR100', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                              shuffle=False, num_workers=4)
    return testloader
