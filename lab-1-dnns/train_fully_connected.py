from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F
from typing import Callable
from sklearn import datasets
from sklearn.model_selection import train_test_split

device = torch.device('cuda')

iris = datasets.load_iris()

preprocessed_features = (iris['data'] - iris['data'].mean(axis=0) / iris['data'])


labels = iris['target']
train_features, test_features, train_labels, test_labels = train_test_split(preprocessed_features, labels, test_size=1/3)

features = {
    'train': torch.tensor(train_features, dtype=torch.float32),
    'test': torch.tensor(test_features, dtype=torch.float32),
}
labels = {
    'train': torch.tensor(train_labels, dtype=torch.long),
    'test': torch.tensor(test_labels, dtype=torch.long),
}

features["train"] = features["train"].to(device)
features["test"] = features["test"].to(device)
labels["train"] = labels["train"].to(device)
labels["test"] = labels["test"].to(device)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, activation_fn = F.relu):

        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_layer_size)
        self.l2 = nn.Linear(hidden_layer_size, output_size)
        self.activation_fn = activation_fn
        
    def forward(self, inputs):
        x = self.l1(inputs)
        x = self.activation_fn(x)
        x = self.l2(x)
        return x

feature_count = 4
hidden_layer_size = 100
class_count = 3


def accuracy(probs, targets):
    """
    Args:
        probs: A float32 tensor of shape ``(batch_size, class_count)`` where each value 
            at index ``i`` in a row represents the score of class ``i``.
        targets: A long tensor of shape ``(batch_size,)`` containing the batch examples'
            labels.
    """
    predicted = torch.argmax(probs, dim=1)
    correct = torch.eq(predicted, targets)
    acc = torch.div(torch.sum(correct), targets.size(dim=0))
    return round(acc.item(), 3)


from torch import optim 

model = MLP(feature_count, hidden_layer_size, class_count)
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.15)
criterion = nn.CrossEntropyLoss()

for epoch in range(0, 100):
    # We compute the forward pass of the network
    logits = model.forward(features['train'])
    # Then the value of loss function 
    loss = criterion(logits,  labels['train'])
    
    # How well the network does on the batch is an indication of how well training is 
    # progressing
    print("epoch: {} train accuracy: {:2.2f}, loss: {:5.5f}".format(
        epoch,
        accuracy(logits, labels['train']) * 100,
        loss.item()
    ))
    
    # Now we compute the backward pass, which populates the `.grad` attributes of the parameters
    loss.backward()
    # Now we update the model parameters using those gradients
    optimizer.step()
    # Now we need to zero out the `.grad` buffers as otherwise on the next backward pass we'll add the 
    # new gradients to the old ones.
    optimizer.zero_grad()
    
# Finally we can test our model on the test set and get an unbiased estimate of its performance.    
logits = model.forward(features['test'])    
test_accuracy = accuracy(logits, labels['test']) * 100
print("test accuracy: {:2.2f}".format(test_accuracy))


