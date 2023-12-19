import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse
import json

def load_data(data_dir):
    """Load and preprocess the data."""
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    # Using the image datasets and the transforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    }

    return dataloaders, image_datasets

def build_model(arch, hidden_units):
    """Build and return the specified pre-trained model with a custom classifier."""
    model = getattr(models, arch)(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier

    return model

def train_model(model, dataloaders, criterion, optimizer, device, epochs=5):
    """Train the model and validate it on the validation set."""
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}.. "
              f"Train loss: {running_loss/len(dataloaders['train']):.3f}")

        validate_model(model, dataloaders['valid'], criterion, device)

def validate_model(model, dataloader, criterion, device):
    """Validate the model on the validation set."""
    model.eval()
    valid_loss = 0
    accuracy = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model.forward(inputs)
            batch_loss = criterion(outputs, labels)

            valid_loss += batch_loss.item()

            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Validation loss: {valid_loss/len(dataloader):.3f}.. "
          f"Validation accuracy: {accuracy/len(dataloader):.3f}")

def save_checkpoint(model, optimizer, arch, hidden_units, epochs, class_to_idx, filepath='checkpoint.pth'):
    """Save the model checkpoint."""
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'arch': arch,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'class_to_idx': class_to_idx
    }

    torch.save(checkpoint, filepath)

def main():
    parser = argparse.ArgumentParser(description='Train a deep learning model on a dataset of images')
    parser.add_argument('data_dir', type=str, help='Path to the dataset directory')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'densenet121'],
                        help='Architecture of the pre-trained model (vgg16 or densenet121)')
    parser.add_argument('--hidden_units', type=int, default=512,
                        help='Number of hidden units in the custom classifier')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth',
                        help='Path to save the trained model checkpoint')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

    dataloaders, image_datasets = load_data(args.data_dir)
    model = build_model(args.arch, args.hidden_units)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    train_model(model, dataloaders, criterion, optimizer, device, epochs=args.epochs)

    model.class_to_idx = image_datasets['train'].class_to_idx
    save_checkpoint(model, optimizer, args.arch, args.hidden_units, args.epochs, model.class_to_idx, args.save_dir)

if __name__ == '__main__':
    main()
