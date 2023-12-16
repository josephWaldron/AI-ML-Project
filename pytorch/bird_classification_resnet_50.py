import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, accuracy_score


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set hyperparameters
num_epochs = 400
batch_size = 64
learning_rate = 0.0001

# Initialize transformations for data augmentation
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=45),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the ImageNet Object Localization Challenge dataset
input_path = "./images/"
full_dataset = datasets.ImageFolder(input_path, transform=transform)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load the ResNet50 model
model = torchvision.models.resnet50(pretrained=True)

# Parallelize training across multiple GPUs
model = torch.nn.DataParallel(model)

# Set the model to run on the device
model = model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model...
epoch_train_loss = []
epoch_test_loss = []
for epoch in range(num_epochs):
    train_loss = []
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in train_loader:
        # Move input and label tensors to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero out the optimizer
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()
      
        train_loss.append(loss.item())
        
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {np.mean(train_loss):.4f}')

    epoch_train_loss.append(np.mean(train_loss))
    
    # predict results for test images:
    model.eval()
    test_loss = []
    for batch_idx, (image, label) in enumerate(test_loader):
        image = image.to("cuda")
        label = label.to("cuda")
        # feed forword
        output = model(image)
        # calculate loss
        loss = criterion(output, label)
        test_loss.append(loss.item())
    print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {np.mean(test_loss):.4f}')

    epoch_test_loss.append(np.mean(test_loss))
    
plt.plot(epoch_train_loss)
plt.plot(epoch_test_loss)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig(f"Loss_{num_epochs}.png")

plt.savefig(f"Test Loss.png")

# Save and Load the Model
model_path = f'bird_classification_{num_epochs}.h5'
torch.save(model.state_dict(), model_path)
model = torch.load(model_path)

model = torchvision.models.resnet50(pretrained=True)
model = torch.nn.DataParallel(model)
model = model.to(device)

model.load_state_dict(torch.load(model_path), strict=False)

model.eval()

def test_model():
    y_test = []

    y_test_predict = []

    model.eval()

    for image, label in test_loader:
        image = image.to("cuda")
        label = label.to("cuda")
        y = model(image).detach().cpu().numpy()[0]
        y_predict = np.argmax(y)
        y_test_predict.append(y_predict)
        y_test.append(label.cpu().numpy()[0])

    print(y_test_predict)
    print(y_test)

    print("acc: ", accuracy_score(y_test, y_test_predict))
test_model()