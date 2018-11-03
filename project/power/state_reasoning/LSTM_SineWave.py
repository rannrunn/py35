import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pylab import grid

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 50
input_size = 1
hidden_size = 64
num_layers = 1
num_classes = 1
batch_size = 64
num_epochs = 20
learning_rate = 0.01

# Sine Wave DataSet
import numpy as np
import pandas as pd


def split_data(data, val_size=0.1, test_size=0.1):
    """
    splits np.array into training, validation and test
    """
    pos_test = int(len(data) * (1 - test_size))
    pos_val = int(len(data[:pos_test]) * (1 - val_size))

    train, val, test = data[:pos_val], data[pos_val:pos_test], data[pos_test:]

    return {"train": train, "val": val, "test": test}


def generate_data(fct, x, time_steps, time_shift):
    """
    generate sequences to feed to rnn for fct(x)
    """
    data = fct(x)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(dict(a=data[0:len(data) - time_shift],
                                 b=data[time_shift:]))
    rnn_x = []
    for i in range(len(data) - time_steps + 1):
        rnn_x.append(data['a'].iloc[i: i + time_steps].as_matrix())
    rnn_x = np.array(rnn_x)

    # Reshape or rearrange the data from row to columns
    # to be compatible with the input needed by the LSTM model
    # which expects 1 float per time point in a given batch
    rnn_x = rnn_x.reshape(rnn_x.shape + (1,))

    rnn_y = data['b'].values
    rnn_y = rnn_y[time_steps - 1:]

    # Reshape or rearrange the data from row to columns
    # to match the input shape
    rnn_y = rnn_y.reshape(rnn_y.shape + (1,))

    return split_data(rnn_x), split_data(rnn_y)


N = sequence_length  # input: N subsequent values == time_steps
M = sequence_length # output: predict 1 value M steps ahead == time_shift
X, Y = generate_data(np.sin, np.linspace(0, 100, 10000, dtype=np.float32), N, M)

# f, a = plt.subplots(3, 1, figsize=(12, 8))
# for j, ds in enumerate(["train", "val", "test"]):
#     a[j].plot(Y[ds], label=ds + ' raw');
# [i.legend() for i in a];

x_train = X["train"]
x_train.shape
y_train = Y["train"]
y_train.shape
# plt.plot(y_train)
# Recurrent neural network - LSTM(many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, dropout=0.05, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
print(model)
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model

x_batch_all = torch.from_numpy(x_train).to(device)
labels_all = torch.from_numpy(y_train).to(device)

total_step = int(len(x_train) / batch_size)
loss_all = []
for epoch in range(num_epochs):
    for i in range(int(len(x_train) / batch_size)):

        # Get a batch of data and labels.
        x_batch = x_batch_all[i * batch_size:i * batch_size + batch_size, :, :]
        labels = labels_all[i * batch_size:i * batch_size + batch_size]
        # Forward pass
        outputs = model(x_batch)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_all.append(loss.cpu().detach().numpy())
        if (i + 1) % 20 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

plt.plot(loss_all)
# Test the model

x_test = X["test"]
x_test = x_test[:400]
# x_test.shape
y_test = Y["test"]
y_test = y_test[:400]
# y_test.shape

x_batch_all_test = torch.from_numpy(x_test).to(device)
labels_all_test = torch.from_numpy(y_test).to(device)

final_outputs = []
test_loss=[]
flag = 0
# Test the model
batch_size = 10
with torch.no_grad(): # torch.no_grad => not to create computation graph, equals to Variable(volatile=True)
    for i in range(int(len(x_test) / batch_size)):
        flag += 1
        # Get a batch of data and labels.
        x_batch = x_batch_all_test[i * batch_size:i * batch_size + batch_size, :, :]
        labels = labels_all_test[i * batch_size:i * batch_size + batch_size]
        # Forward pass
        outputs = model(x_batch)
        outputs_data = outputs.squeeze().cpu().detach()
        if flag == 1:
            final_outputs = outputs_data
        else:
            final_outputs = torch.cat((final_outputs,outputs_data),0)

        loss = criterion(outputs, labels)

        test_loss.append(loss.cpu().detach().numpy())
        if (i + 1) % 20 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

final_pred = np.array(final_outputs)
plt.plot(test_loss)

plt.plot(final_pred, label='predicted')
plt.plot(y_test, label='actual')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
grid(True)
# plt.plot(final_pred, label='predicted')
# plt.plot(y_test[5:], label='actual')

# plt.show()