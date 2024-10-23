import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def function(x):
    return torch.mean(x * x * (x < 0.5), dim=1) + torch.mean((-x+3.4) * (x > 0.5), dim=1)

D = 30
d = 10

# Define the deep regression model with more layers
class DeepRegressionModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DeepRegressionModel, self).__init__()

        # Input layer
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU()]

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), nn.ReLU()])

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
# Hyperparameters
input_size = D  # Change this based on your input features
hidden_sizes = [D, D]  # Adjust the number of hidden layers and units as needed
output_size = 1  # Regression has one output
learning_rate = 0.001
epochs = 100
b_size = 100


device = torch.device('cuda')
n_rep = 20
n_m = 10
n_n = 10
m_test_non = 100
n_test = 100
participating_error = torch.empty((n_m, n_n, n_rep)).to(device)
non_participating_error = torch.empty((n_m, n_n, n_rep)).to(device)


for ct_m in range(n_m):
    for ct_n in range(n_n):
        for rep in range(n_rep):
            m = (ct_m+1) * 20
            n = (ct_n+1) * 20
            N_train = m * n
            N_test = m * n_test
            N_test_non = m_test_non * n_test

            train_batch_size = n
            test_batch_size = n_test
            
            # Make the Data
            
            Theta_0 = torch.rand(m, D).to(device)
            Theta = Theta_0.repeat(n,1)
            Theta_test = Theta_0.repeat(n_test,1)

            Theta_non = torch.rand(m_test_non, D).to(device)
            Theta_non = Theta_non.repeat(n_test,1)

            X_train = torch.rand(N_train, D).to(device)
            X_train[:, (d + 1):D] = 0
            Theta[:, (d + 5):D] = 0
            y_train = torch.zeros(m * n).to(device)

            X_test = torch.rand(N_test, D).to(device)
            X_test[:, (d + 1):D] = 0
            Theta_test[:, (d + 5):D] = 0
            y_test = torch.zeros(m * n_test).to(device)

            X_test_non = torch.rand(N_test_non, D).to(device)
            X_test_non[:, (d + 1):D] = 0
            Theta_non[:, (d + 5):D] = 0
            y_test_non = torch.zeros(m_test_non * n_test).to(device)

            X_train = X_train + Theta
            X_test = X_test + Theta_test
            X_test_non = X_test_non + Theta_non


            y_train = function(X_train) + torch.rand(N_train).to(device) * 0.1
            y_test = function(X_test) #+ torch.rand(N_test).to(device) * 0.1
            y_test_non = function(X_test_non) #+ torch.rand(N_test_non).to(device) * 0.1   
            
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle = True)
            test_dataset = TensorDataset(X_test, y_test)

            test_loader = DataLoader(test_dataset, batch_size=b_size, shuffle = True)
            test_dataset_non = TensorDataset(X_test_non, y_test_non)
            test_loader_non = DataLoader(test_dataset_non, batch_size=b_size, shuffle = True)    
            
            model = DeepRegressionModel(input_size, hidden_sizes, output_size)
            model.to(device)
            
            # Training
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            for epoch in range(epochs):
                for inputs, targets in train_loader:
                    targets = targets.view(-1, 1)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                
            # Testing
            with torch.no_grad():
                total_loss = 0
                for inputs, targets in test_loader:
                    outputs = model(inputs)
        
                    # Flatten the target tensor if needed
                    targets = targets.view(-1, 1)
        
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()

            participating_error[ct_m, ct_n, rep] = total_loss / len(test_loader)
            with torch.no_grad():
                total_loss = 0
                for inputs, targets in test_loader_non:
                    outputs = model(inputs)
        
                    # Flatten the target tensor if needed
                    targets = targets.view(-1, 1)
        
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()

            non_participating_error[ct_m, ct_n, rep] = total_loss / len(test_loader_non)  
            
        print(f'm iteration = {ct_m+1}, n iteration = {ct_n+1}')
        
                   
torch.save(participating_error, 'e2_p_error_30_10.pt')
torch.save(non_participating_error, 'e2_np_error_30_10.pt')