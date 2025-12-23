import torch
import torch.nn as nn
import torch.nn.functional as F

# Задаем веса и данные
inputs = torch.tensor([1.0, 2.0, 3.0])
weights_input_hidden = torch.tensor([[0.1, 0.2, 0.3],
                                     [0.4, 0.5, 0.6],
                                     [0.7, 0.8, 0.9]])
weights_hidden_output = torch.tensor([[0.2, 0.4, 0.6],
                                     [0.8, 0.5, 0.2]])

# Задаем биасы
bias_input_hidden = torch.tensor([0.1, 0.2, 0.3])
bias_hidden_output = torch.tensor([0.2, 0.1])

# Определяем нейронную сеть
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(3, 3)
        self.output = nn.Linear(3, 2)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x

# Создаем экземпляр нейронной сети
model = NeuralNetwork()

# Проход через нейронную сеть
output = model(inputs)
print(output)