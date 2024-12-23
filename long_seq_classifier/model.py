import torch
import torch.nn as nn


class DocumentClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(DocumentClassifier, self).__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, document_representation):
        logits = self.classifier(document_representation)
        return logits
