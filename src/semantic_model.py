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
        self.mapping_list = None
        
        for _ in range(num_layers):
            self.layers.append(ResidualBlock(hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        
        return x

    def load_semantic_probs(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(torch.cuda.is_available)
        bert_name = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        bert = BertModel.from_pretrained(bert_name).to(device)
        bert.eval()
        self.bert = bert
        
        semantic_model = SemanticModel().to(device)
        state = torch.load("model/semantic_mapping_model.pth", map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            semantic_model.load.state_dict(state["model_state_dict"])
        else:
            semantic_model.load_state_dict(state)
        self.model = semantic_model.eval()


    def build_mapping_list(self, save_path="model/semantic_mapping_list.pt"):
        if self.bert is None or self.model is None:
            raise RuntimeError("Call load_semantic_probs() before build_mapping_list().")

        device = next(self.model.parameters()).device

        print("Building mapping list… this may take 1–2 minutes…")

        # (A) Get BERT word embeddings [vocab, 768]
        emb = self.bert.embeddings.word_embeddings.weight.to(device)

        # (B) Push through semantic model → [vocab, 384]
        with torch.no_grad():
            self.mapping_list = self.model(emb).cpu()

        # Save for future use
        torch.save(self.mapping_list, save_path)

        print(f"✓ Mapping list built. Shape = {self.mapping_list.shape}")
        print(f"✓ Saved to {save_path}")

        return self.mapping_list


    @torch.no_grad()
    def encode_text(self, text: str, max_length: int = 128):
        """
        Text -> BERT-Embedding -> SemanticModel -> 384-D Vector
        """
        assert self.tokenizer is not None and self.bert is not None, \
            "Bitte vorher load_semantic_probs() aufrufen."

        device = next(self.parameters()).device

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        ).to(device)

        # BERT: [batch, seq, hidden]
        outputs = self.bert(**inputs)
        # CLS-Token-Embedding nehmen
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [1, 768]

        # SemanticModel anwenden
        z = self(cls_emb)  # [1, 384]
        return z  # Tensor


    def calc_cos_sim(self, input1, input2):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        return cos(self.encode_text(input1), self.encode_text(input2))

