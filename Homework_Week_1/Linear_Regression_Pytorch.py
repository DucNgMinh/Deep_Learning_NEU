import torch
import torch.nn as nn

class Linear_Regression():
    def __init__(self, num_features):
        self.num_features = num_features
        self.w = torch.randn(self.num_features, 1, requires_grad = True, dtype= torch.float64)
        self.b = torch.randn( 1, requires_grad = True, dtype= torch.float64)

    def fit(self, x, y, num_epochs= 100,  lr= 0.0001):
        self.losses = []
        for i in range(num_epochs):
            y_hat = self.predict(x)

            loss = torch.mean((y_hat - y) ** 2)

            loss.backward()
            # Adjust weights & reset gradients
            self.losses.append(loss.item())

            with torch.no_grad():
                self.w -= lr * self.w.grad 
                self.b -= lr * self.b.grad 

                self.w.grad.zero_()
                self.b.grad.zero_()

            if (i + 1) % 10 == 0:
                print(f'epoch {i + 1}: loss = {loss.item():.2f}')

    def loss(self):
        return self.losses

    def predict(self, x):
        return x @ self.w.t() + self.b
    