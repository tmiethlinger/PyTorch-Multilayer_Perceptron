import numpy as np

import torch
from torch import nn
from torch import optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiLayerPerceptron(nn.Module):

    def __init__(self, layer_widths, activation_function, seed=1):
        super(MultiLayerPerceptron, self).__init__()

        # set random seeds for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        self.input_size = layer_widths[0]
        self.output_size = layer_widths[-1]

        self.layer_widths = layer_widths

        self.activation_function = activation_function

        self.linear_layers = nn.ModuleList([nn.Linear(w[0], w[1]) for w in zip(self.layer_widths[:-1], self.layer_widths[1:])])

        self.activation_layers = nn.ModuleList([self.activation_function for a in range(len(self.layer_widths) - 1)])


    def forward(self, x):

        # State s is input data x
        s = x

        # Iterate layers
        for linear, activation in zip(self.linear_layers[:-1], self.activation_layers):
            s = activation(linear(s))

        # Output y
        y = self.linear_layers[-1](s)

        return y


    def train_model(self, loss, lr, batch_size, num_epochs, train_dataset, test_dataset):

        # Loss and optimizer
        self.loss = loss
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Create PyTorch DataLoader for train and test data
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = len(list(test_dataset)), shuffle = False, drop_last = True)

        # set our model in the training mode
        self.train()

        print("Epoch MSELoss(train) MSELoss(test)")
        for epoch in range(num_epochs):

            # Compute loss for train set,
            # update weights
            epoch_loss_train = 0
            for i, batch_sample in enumerate(train_loader):

                x_batch, y_batch = batch_sample

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                prediction = self(x_batch)

                l = self.loss(prediction, y_batch)
                epoch_loss_train += l.item()

                self.zero_grad()
                l.backward()
                self.optimizer.step()
            epoch_loss_train /= len(train_loader)

            # Compute loss for test set
            epoch_loss_test = 0
            for i, batch_sample in enumerate(test_loader):

                x_batch, y_batch = batch_sample

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                prediction = self(x_batch)

                l = self.loss(prediction, y_batch)
                epoch_loss_test += l.item()
            epoch_loss_test /= len(test_loader)

            if epoch % 100 == 0:
                print(str(epoch) + " " + "{:.5f}".format(epoch_loss_train) + " " + "{:.5f}".format(epoch_loss_test))


    def save_model(self, path):
        torch.save(self.state_dict(), path + ".pt")


    def load_model(self, path):
        self.load_state_dict(torch.load(path + ".pt"))
        self.eval()
        return self
