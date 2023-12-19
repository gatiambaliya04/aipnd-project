import torch
from torch import nn
from torchvision import models, transforms
import argparse
from PIL import Image
import json

def process_image(image_path):
    """Process an image for prediction."""
    image = Image.open(image_path)
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = image_transform(image)
    return image.numpy()

def load_checkpoint(filepath):
    """Load a model checkpoint."""
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, checkpoint['hidden_units']),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(checkpoint['hidden_units'], 102),
        nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def predict(image_path, model, topk=5):
    """Predict the class of an image using a trained model."""
    model.eval()
    image = process_image(image_path)
    image = torch.from_numpy(image).float()
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model.forward(image)

    probabilities, classes = torch.topk(torch.exp(output), topk)
    probabilities = probabilities.cpu().numpy().squeeze()
    classes = classes.cpu().numpy().squeeze()

    return probabilities, classes

def main():
    parser = argparse.ArgumentParser(description='Predict the class of an image using a trained deep learning model')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='Path to a JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for prediction')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

    model = load_checkpoint(args.checkpoint)
    model.to(device)

    # Process image and make prediction
    probabilities, classes = predict(args.image_path, model, topk=args.top_k)

    # Load category names from JSON file
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Map classes to category names
    class_names = [cat_to_name[str(cls)] for cls in classes]

    # Print top K classes and their associated probabilities
    for i in range(args.top_k):
        print(f"Top {i + 1}: {class_names[i]} - Probability: {probabilities[i]:.4f}")

if __name__ == '__main__':
    main()
