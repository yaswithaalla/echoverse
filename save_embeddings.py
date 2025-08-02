import torch

# SpeechT5 expects embeddings of shape (1, 512)
embedding_size = 512  

voices = ["Lisa", "Michael", "Allison", "Kate"]

print("Generating speaker embeddings for EchoVerse...")
print("=" * 50)

for voice in voices:
    emb = torch.randn(1, embedding_size)  # Random embedding (works fine for demo)
    torch.save(emb, f"{voice.lower()}_emb.pt")
    print(f"✅ Created embedding for {voice} -> {voice.lower()}_emb.pt")

print("=" * 50)
print("All embeddings created successfully!")
print("\nTo use different voices in production, replace these with:")
print("- Real speaker embeddings from audio samples")
print("- Pre-trained voice embeddings from datasets")
print("- Custom voice cloning embeddings")
