import torch
import torch.nn as nn
import torch.quantization as quant
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from sim.circuits_obj import *
from sim.SEC import *

def float32_to_radix_arr(f32_tensor, nbits):
    negative_mask = torch.sign(f32_tensor)
    f32_tensor_abs = torch.abs(f32_tensor)
    assert torch.all(f32_tensor_abs <= 1.0) #Alg doesn't work if weights aren't in range -1 to 1
    f32_tensor_abs_ = torch.clone(f32_tensor_abs)
    val = torch.empty_like(f32_tensor, dtype=torch.float32).fill_(0.5)
    result = torch.zeros_like(f32_tensor, dtype=torch.bool)
    result = torch.cat(nbits*[result.unsqueeze(0)])
    for i in range(nbits):
        gt_val = f32_tensor_abs_ - val >= 0.0
        result[i] = gt_val
        f32_tensor_abs_ -= gt_val * val
        val /= 2
    return result, (f32_tensor_abs - f32_tensor_abs_) * negative_mask

class CNN(nn.Module):
    def __init__(self, cifar=True, quantize=False):
        super().__init__()
        self.quantize = quantize
        if cifar:
            self.conv1 = nn.Conv2d(3, 6, 5) #32 --> 28
            self.pool = nn.MaxPool2d(2, 2) #28 --> 14
            self.conv2 = nn.Conv2d(6, 16, 5) #14 --> 10
            self.fc1 = nn.Linear(16 * 5 * 5, 76)
        else:
            self.conv1 = nn.Conv2d(1, 6, 5) #28 --> 24
            self.pool = nn.MaxPool2d(2, 2) #24 --> 12
            self.conv2 = nn.Conv2d(6, 16, 5) #12 --> 8
            self.fc1 = nn.Linear(16 * 4 * 4, 76)
        self.fc2 = nn.Linear(76, 48)
        self.fc3 = nn.Linear(48, 10)
        if quantize:
            self.quant = quant.QuantStub()
            self.dequant = quant.DeQuantStub()

    def forward(self, x):
        if self.quantize:
            x = self.quant(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.quantize:
            x = self.dequant(x)
        return x

    def sc_quantize(self, nbits):
        state_dict = self.state_dict()
        self.sc_quant_weights = {}
        self.nbits = nbits
        for name, param in state_dict.items():
            bit_arr, transformed_param = float32_to_radix_arr(param, nbits)
            self.sc_quant_weights[name] = transformed_param
            param.copy_(transformed_param)

    def sc_circuit(self):

        #Normal conv/fc+ReLU layers
        sc_layers = {}
        for name, param in self.sc_quant_weights.items():
            if 'conv' in name:
                if 'weight' in name: #weights
                    sc_layers[name] = []
                    for channel in range(param.shape[1]):
                        weights = torch.flatten(param[channel, :, :, :], start_dim=1, end_dim=2)
                        sc_layers[name].append([PARALLEL_CONST_MUL(weights, self.nbits, bipolar=False, reuse=True), ])
                else: #biases
                    pass
                    #current_channels[channel]
                    #Most of the code will go here

                    #Add ReLU
                    
            elif 'fc' in name:
                pass
            else:
                raise TypeError("Unknown layer type")

        #Add max pooling to the correct locations

        return SeriesCircuit(sc_layers)

def CNN_classifier_main(cifar=True, train=True, test=True, quantize=False, sc_quantize=False, nbits=4):
    if cifar:
        PATH = './CIFAR_net.pth'
        norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    else:
        PATH = './MNIST_net.pth'
        norm = transforms.Normalize((0.1307,), (0.3081,))
    batch_size = 4
    num_epochs = 5
    transform = transforms.Compose([transforms.ToTensor(), norm])

    #Train
    if train:
        if cifar:
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        else:
            trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        cnn = CNN(cifar=cifar, quantize=quantize)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = cnn(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
        torch.save(cnn.state_dict(), PATH)

    if test:
        if cifar:
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        else:
            testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        cnn = CNN(cifar=cifar, quantize=quantize)
        cnn.load_state_dict(torch.load(PATH))
        cnn.eval()
        if quantize:
            cnn.qconfig = quant.get_default_qconfig('fbgemm')
            quant.prepare(cnn, inplace=True)
            for data in testloader:
                images, _ = data
                cnn(images) #calibration for quantization
            quant.convert(cnn, inplace=True)
            print("quantized")
        elif sc_quantize:
            cnn.sc_quantize(nbits)
            test_circ = cnn.sc_circuit()

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = cnn(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the test images: {100 * correct // total} %')
        return 100 * correct / total
    
if __name__ == "__main__":
    #for i in range(1, 9):
    CNN_classifier_main(cifar=False, train=False, sc_quantize=True, nbits=4)