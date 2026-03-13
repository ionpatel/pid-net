"""Quick generation test — run alongside training."""
import torch
from ionbrain import create_ionbrain

model = create_ionbrain("small")
ckpt = torch.load("checkpoints/ionBrain-small_best.pt", map_location="cpu", weights_only=True)
model.load_state_dict(ckpt)
model.eval()

prompts = [
    "Once upon a time",
    "The king said to",
    "She walked into the",
    "ABCDEFG",
    "1 + 1 = ",
]

for p in prompts:
    x = torch.tensor([[ord(c) for c in p]])
    with torch.no_grad():
        gen = model.generate(x, max_new=150, temperature=0.8, top_k=40)
    text = "".join(chr(min(b, 127)) for b in gen[0].tolist())
    print()
    print("=" * 50)
    print("Prompt:", p)
    print("Output:", text)
