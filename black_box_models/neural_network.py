import torch
import torch.nn as nn

class NeuralNetModel (nn.Module):
    def __init__(self, embeddings, hidden_size, embeddings_size, num_layers, num_classes = 3):
        self.premise_layer = nn.Sequential(
            nn.Embedding.from_pretrained(embeddings),
            nn.LSTM(input_size = embeddings_size, hidden_size = hidden_size, num_layers=num_layers, batch_first=True)
        )
        self.hypothesis_layer = nn.Sequential(
            nn.Embedding.from_pretrained(embeddings),
            nn.LSTM(input_size = embeddings_size, hidden_size = hidden_size, num_layers=num_layers, batch_first=True)
        )
        self.model_layers = nn.Sequential(
            nn.Linear(hidden_size*2, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, num_classes)
        )

    def forward(self, premise_batch, hypothesis_batch):
        premise_output = self.premise_layer(premise_batch)
        hypothesis_output = self.hypothesis_layer(hypothesis_batch)
        print(premise_output.shape)
        print(hypothesis_output.shape)
