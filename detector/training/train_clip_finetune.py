from transformers import CLIPProcessor, CLIPModel
import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Define the training parameters
batch_size = 16
num_epochs = 5
learning_rate = 5e-5

# Define the prompts for the specific scenarios
prompts = [
    "A person writing notes",
    "A person reading",

    "A person sitting",
    "A person drinking",
    "A person looking at a watch",
    "A person yawning",

    "A person using a phone",
    "A person looking away",
    "A person stretching",
    "A person slouching",
    "A person fidgeting",
    "A person resting their head on their hand",
    "A person looking around",

    "An empty chair",
    "A person leaving",

    "A person sleeping",
]

# Tokenize the prompts
text_inputs = processor(text=prompts, return_tensors="pt",
                        padding=True, truncation=True).to(device)

# Define the dataset and dataloader
data_dir = os.path.join(os.path.dirname(__file__), '../data/processed')
dataset = datasets.ImageFolder(data_dir, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for images, _ in tqdm(dataloader):
        images = images.to(device)

        # Process images and compute loss
        outputs = model(images, text_inputs['input_ids'], return_loss=True)
        loss = outputs.loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save the fine-tuned model
model.save_pretrained(os.path.join(
    os.path.dirname(__file__), '../models/clip_finetuned'))
