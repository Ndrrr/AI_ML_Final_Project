import torch
import torch.nn as nn
import torchvision.models as model


class ExModel(nn.Module):
    
    def __init__(self):
        super().__init__()
                
        self.resnet18 = model.resnet18(pretrained=False)
        self.resnet18 = torch.nn.Sequential(*(list(self.resnet18.children())[:-1]))    
        
        self.resnet18classifier = torch.nn.Linear(512, 58) #TODO Change this classifier according to your application!


        self.vgg16 = model.vgg16(pretrained=False)

        num_features = self.vgg16.classifier[6].in_features
        features = list(self.vgg16.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, 58)]) # Add our layer with 4 outputs
        self.vgg16.classifier = nn.Sequential(*features)

    def forward(self, image):
        return self.forward_resnet(image)
    
    def forward_resnet(self, image):
        # Get predictions from ResNet18
        resnet_pred = self.resnet18(image).squeeze()
        out = self.resnet18classifier(resnet_pred)
        
        return out
    
    def forward_vgg(self, image):
        return self.vgg16(image)
    