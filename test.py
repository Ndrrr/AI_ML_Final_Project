import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from datasets.dataset_retrieval import custom_dataset  # Assuming you have a custom dataset class
from models.example_model import ExModel  # Assuming you have a model class
from torchmetrics import F1Score
import tqdm


# Define transformations for the test data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create a test dataset and dataloader
test_dataset = custom_dataset(mode='test', transforms=transform)
test_loader = DataLoader(test_dataset, batch_size=58, shuffle=False)

# Load the trained model
model = ExModel()  # Instantiate your model class
model.load_state_dict(torch.load('checkpoints/sgd_vgg_0_01.pth', map_location='cpu')['state_dict'])
model.eval()  # Set the model to evaluation mode

# Define a device for inference (CPU or GPU)
device = torch.device('cpu')

# Move the model to the device
model.to(device)

f1score = 0
f1 = F1Score(num_classes=58, task = 'multiclass')
data_iterator = enumerate(test_loader)  # take batches
pred_list = []
label_list = []

with torch.no_grad():
    model.eval()  # switch model to evaluation mode
    tq = tqdm.tqdm(total=len(test_loader))
    tq.set_description('Test:')

    total_loss = 0

    for _, batch in data_iterator:
        # forward propagation
        image, label = batch
        image = image.to(device)
        label = label.to(device)
        pred = model(image)

        pred = pred.softmax(dim=1)
        
        pred_list.extend(torch.argmax(pred, dim =1).tolist())
        label_list.extend(torch.argmax(label, dim =1).tolist())

        print(pred)
        print(label)
        # compare
        a = torch.argmax(pred, dim =1).tolist()
        b = torch.argmax(label, dim =1).tolist()

        for i in range(len(a)):
            if a[i] != b[i]:
                print("Predicted: ", a[i], "Actual: ", b[i])

        tq.update(1)


f1score = f1(torch.tensor(pred_list), torch.tensor(label_list))

tq.close()
print("F1 score: ", f1score)

