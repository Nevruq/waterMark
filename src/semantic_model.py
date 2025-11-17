import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AutoTokenizer

# Part of the Code from https://github.com/yepengliu/adaptive-text-watermark


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        out = out + x 
        return out

class SemanticModel(nn.Module):
    def __init__(self, num_layers=2, input_dim=768, hidden_dim=512, output_dim=384):
        super(SemanticModel, self).__init__()
        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        self.tokenizer = None
        self.bert = None
        self.model = None
        
        for _ in range(num_layers):
            self.layers.append(ResidualBlock(hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        
        return x

    def load_semantic_probs(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # BERT laden – muss derselbe sein wie beim Training!
        bert_name = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        bert = BertModel.from_pretrained(bert_name).to(device)
        bert.eval()
        self.bert = bert

        # Dein SemanticModel + Gewichte laden
        semantic_model = SemanticModel().to(device)
        state = torch.load("model/semantic_mapping_model.pth", map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            semantic_model.load_state_dict(state["model_state_dict"])
        else:
            semantic_model.load_state_dict(state)
        self.model = semantic_model.eval()

    def calc_cos_sim(self, input1, input2):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(self.encode_text(input1), self.encode_text(input2))

