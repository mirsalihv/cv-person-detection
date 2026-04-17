import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from dataset import PersonDataset


def get_model():
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    num_classes = 2  # background + person
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # DEVICE CHECK
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available ")
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)
    else:
        device = torch.device("cpu")
        print("CUDA NOT available — using CPU")

    # DATASET
    dataset = PersonDataset(
        "data/images",
        "data/labels",
        transforms=T.ToTensor()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )

    # MODEL
    model = get_model()
    model.to(device)

    # OPTIMIZER
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    print("Starting training...")

    # TRAIN LOOP
    for epoch in range(3):
        total_loss = 0

        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # SAVE MODEL
    torch.save(model.state_dict(), "models/model.pth")
    print("Training complete. Model saved.")


if __name__ == "__main__":
    main()