import torch

# SpeechT5 expects embeddings of shape (1, 512)
embedding_size = 512  

voices = ["Lisa", "Michael", "Allison", "Kate"]

for voice in voices:
    emb = torch.randn(1, embedding_size)  # Random embedding (works fine for demo)
    torch.save(emb, f"{voice.lower()}_emb.pt")
    print(f"Created embedding for {voice} -> {voice.lower()}_emb.pt")
