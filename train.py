import torch
import torch.nn as nn
from models.fire_detection_model import load_pretrained_model
from utils.datasets import create_dataloader
from utils.general import check_dataset, check_img_size
from utils.torch_utils import select_device, init_seeds
import yaml

# Configuration
data_cfg = 'data/fire_detection.yaml'
cfg = 'models/yolov5s.yaml'
img_size = 640
batch_size = 16
epochs = 50
weights = 'yolov5s.pt'
device = select_device('')
init_seeds(42)

# Dataset
with open(data_cfg) as f:
    data_dict = yaml.load(f, Loader=yaml.SafeLoader)
train_path = data_dict['train']
val_path = data_dict['val']

# Dataloader
dataloader, dataset = create_dataloader(train_path, img_size, batch_size, 16, hyp=None, augment=True, cache=False, rect=False, rank=-1, world_size=1, workers=8)
val_loader = create_dataloader(val_path, img_size, batch_size, 16, hyp=None, augment=False, cache=False, rect=True, rank=-1, world_size=1, workers=8)[0]

# Model
model = load_pretrained_model()
model.to(device)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    for i, (imgs, targets, paths, _) in enumerate(dataloader):
        imgs = imgs.to(device).float() / 255.0
        targets = targets.to(device)

        pred = model(imgs)
        loss = criterion(pred, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}")

    # Evaluate the model
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, targets, paths, shapes in val_loader:
            imgs = imgs.to(device).float() / 255.0
            targets = targets.to(device)

            pred = model(imgs)
            val_loss += criterion(pred, targets).item()

    print(f"Epoch {epoch}, Validation Loss: {val_loss / len(val_loader)}")
