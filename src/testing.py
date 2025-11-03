from transformers import AutoTokenizer

# choose any pretrained model from the Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained("gpt2")

texts = ["Hello world!", "How are you today?"]
encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
print(encodings["input_ids"].shape)

print(tokens)