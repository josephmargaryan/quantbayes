import torch
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from torch.optim import Adam
import torch.nn.functional as F
import pandas as pd


# Function to generate fake data
def generate_fake_data(num_companies=10, num_features=2):
    """
    Generate fake data simulating company sentiment and stock prices.

    Args:
        num_companies (int): Number of companies to simulate.
        num_features (int): Number of features per company.

    Returns:
        DataFrame: Simulated data in the required format.
    """
    companies = [f"Company_{i}" for i in range(num_companies)]
    predicted_labels = np.random.choice(
        ["Positive", "Neutral", "Negative"], size=num_companies
    )
    close_prices = np.random.uniform(50, 500, size=num_companies)

    data = {
        "Company": companies,
        "predicted_labels": predicted_labels,
        "Close": close_prices,
    }

    return pd.DataFrame(data)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.fc = Linear(out_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x


def main():
    fake_data = generate_fake_data()

    label_encoder = LabelEncoder()
    fake_data["predicted_labels"] = label_encoder.fit_transform(
        fake_data["predicted_labels"]
    )

    node_features = fake_data[["predicted_labels", "Close"]].values
    node_features = torch.tensor(node_features, dtype=torch.float)

    num_nodes = len(fake_data)
    adjacency_matrix = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)

    edges = np.array(np.where(adjacency_matrix == 1))
    edge_index = torch.tensor(edges, dtype=torch.long)

    graph_data = Data(x=node_features, edge_index=edge_index)

    in_channels = graph_data.x.size(1)
    hidden_channels = 16
    out_channels = 8
    num_classes = len(label_encoder.classes_)
    model = GCN(in_channels, hidden_channels, out_channels, num_classes)

    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        out = model(graph_data)
        target = graph_data.x[:, 0].long()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    model.eval()

    with torch.no_grad():
        out = model(graph_data)
        predicted_indices = out.argmax(dim=1)

    predicted_labels = label_encoder.inverse_transform(predicted_indices.cpu().numpy())

    companies = fake_data["Company"].values
    for company, label in zip(companies, predicted_labels):
        print(f"Company: {company}, Predicted Sentiment: {label}")

    G = to_networkx(graph_data, to_undirected=True)

    nx.set_node_attributes(
        G,
        {i: label for i, label in enumerate(predicted_labels)},
        name="predicted_sentiment",
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels={i: companies[i] for i in range(len(companies))},
        node_color=predicted_indices.cpu().numpy(),
        cmap="viridis",
        node_size=500,
        font_size=10,
        ax=ax,
    )
    plt.title("Graph of Companies with Predicted Sentiments")
    plt.colorbar(
        plt.cm.ScalarMappable(cmap="viridis"), label="Predicted Sentiment", ax=ax
    )
    plt.show()


if __name__ == "__main__":
    main()
