import torch
import argparse
from PIL import Image
from torchvision import transforms
from config import Config
from models.deepfake_detector import DeepFakeDetector

def predict_image(model, image_path, device, threshold=0.45):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        prob = torch.sigmoid(logits).item()

    label = "FAKE" if prob > threshold else "REAL"
    return label, prob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--weights", type=str, default="weights/best_model.pth", help="Path to weights")
    parser.add_argument("--threshold", type=float, default=0.45, help="Classification threshold")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepFakeDetector().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    label, prob = predict_image(model, args.image, device, args.threshold)
    print(f"Result: {label} (confidence: {prob:.4f})")
