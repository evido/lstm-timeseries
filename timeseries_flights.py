import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader


class FlightDataset(Dataset):
    def __init__(self, passengers):
        self.train_window = 12
        self.passengers = passengers

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.passengers_normalized = self.scaler.fit_transform(self.passengers.reshape(-1, 1))
        self.passengers_normalized = torch.FloatTensor(self.passengers_normalized)
        self.passengers_sequences = self._create_inout_sequences(self.passengers_normalized)

    def _create_inout_sequences(self, input_data):
        inout_seq = []
        length = len(input_data)
        for i in range(length - self.train_window):
            train_seq = input_data[i:i + self.train_window]
            train_label = input_data[i + self.train_window:i + self.train_window + 1]
            inout_seq.append((train_seq, train_label))
        return inout_seq

    def __getitem__(self, item):
        return self.passengers_sequences[item]

    def __len__(self):
        return len(self.passengers_sequences)


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, num_layers=1):
        super().__init__()

        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        self.input_size = input_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, self.hidden_layer_size, num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_layer_size, self.output_size)

        self.hidden_cell = None

    def forward(self, input_seq):
        # model_input_seq = input_seq.view(len(input_seq), 1, -1)
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(lstm_out.view(input_seq.shape[0], input_seq.shape[1], -1))
        return predictions[:, -1]

    def reset(self, batch_size, device):
        self.hidden_cell = (torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(device),
                            torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(device))


def build_dataset(test_data_size=12):
    flights = sns.load_dataset("flights")
    all_data = flights["passengers"].values.astype(float)
    train_data = all_data[:-test_data_size]
    test_data = all_data[-test_data_size:]
    return FlightDataset(train_data), FlightDataset(test_data)


def train(train_dataset, epochs=150, batch_size=12, lr=0.001, device="cuda"):
    model = LSTM().to(device)
    loss_function = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch_index in range(epochs):
        for _, sample in enumerate(loader):
            seq, labels = sample
            optimizer.zero_grad()
            model.reset(batch_size, device)
            y_pred = model(seq.to(device))

            single_loss = loss_function(y_pred.view(batch_size, 1, 1), labels.to(device))
            single_loss.backward()
            optimizer.step()

        if epoch_index % 25 == 1:
            print(f"epoch: {epoch_index:3} loss: {single_loss.item():10.8f}")

    print(f"epoch: {epoch_index:3} loss: {single_loss.item():10.10f}")

    return model


def validate(train_dataset, model, train_window=12, fut_pred=12, device="cuda"):
    model.eval()

    test_seq, test_label = train_dataset[-1]
    test_seq = test_seq.view(-1).tolist()

    for _ in range(fut_pred):
        seq = torch.FloatTensor(test_seq[-train_window:]).to(device)
        with torch.no_grad():
            model.reset(1, device)
            prediction = model(seq.view(1, train_window, 1))
            test_seq.append(prediction[0].cpu())

    predictions = np.array(test_seq[-fut_pred:]).reshape(-1, 1)
    actual_predictions = train_dataset.scaler.inverse_transform(predictions)
    actual_predictions = actual_predictions.reshape(-1, 1)

    return actual_predictions


def display_predictions(train_dataset, validation_dataset, predictions):
    x = np.arange(132, 144, 1)

    plt.title("Month vs Passenger")
    plt.ylabel("Total Passengers")
    plt.grid(True)
    plt.autoscale(axis="x", tight=True)
    plt.plot(train_dataset.passengers.tolist() + validation_dataset.passengers.tolist())
    plt.plot(x, predictions)
    plt.savefig("passengers.png")


def main(**kwargs):
    train_dataset, validate_dataset = build_dataset()
    model = train(train_dataset, **kwargs)
    predictions = validate(train_dataset, model)
    display_predictions(train_dataset, validate_dataset, predictions)


if __name__ == "__main__":
    main(device="cuda", lr=0.00003, epochs=5000)
