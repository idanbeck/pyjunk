import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np

class SimpleTorchSolver():
    def __init__(self, model, params, *args, **kwargs):
        super(SimpleTorchSolver, self).__init__(*args, *kwargs)
        self.params = params
        self.lr = params['lr']
        self.epochs = params['epochs']
        self.grad_clip = params.get('grad_clip')    # If None won't do anything below
        self.model = model

        self.strOptimizer = params['strOptimizer']
        if(self.strOptimizer == 'Adam'):
            self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr = self.lr)

    def train(self, train_data_images):
        self.model.train()
        training_losses = []

        for image in train_data_images:
            loss = self.model.loss_with_image(image)

            self.optimizer.zero_grad()
            loss.backward()

            if(self.grad_clip):
                nn.utis.clip_grad_norm(self.model.parameters(), self.grad_clip)

            self.optimizer.step()
            training_losses.append(loss.item())

        return training_losses


    def test(self, test_data_images):
        self.model.eval()
        loss = 0.0

        with torch.no_grad():
            for image in test_data_images:
                loss += self.model.loss_with_image(image)

            loss /= len(test_data_images)

        return loss.item()

    def train_for_epochs(self, train_data_images, test_data_images):
        training_losses = []
        test_losses = []

        for epoch in range(self.epochs):
            train_losses = self.train(train_data_images)
            training_losses.extend(train_losses)

            test_loss = self.test(test_data_images)
            test_losses.append(test_loss)

            print(f'Epoch {epoch}, Test loss {test_loss:.4f}')

        return training_losses, test_losses

    def train_and_visualize(self, train_data_images, test_data_images, strTitle="Train, Test Loss Plot"):
        # Train and evaluate the model
        training_losses, test_losses = self.train_for_epochs(train_data_images, test_data_images)

        plt.figure()
        n_epochs = len(test_losses) - 1
        x_train = np.linspace(0, n_epochs, len(training_losses))
        x_test = np.arange(n_epochs + 1)

        plt.plot(x_train, training_losses, label='train loss')
        plt.plot(x_test, test_losses, label='test loss')
        plt.legend()
        plt.title(strTitle)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # Visualize image
        testImage = test_data_images[0]
        imagePassThru = self.model.forward_with_image(testImage)
        imagePassThru.visualize()