from torchvision import transforms
from torch.utils import data
from torch.utils.data import Dataset
from PIL import Image
from BalancedClass import make_weights_for_balanced_classes

import os
import torch


class ImageTextData(Dataset):
    
    def __init__(self, data, transform = False):
        
        self.data = data
        self.transform = transform
        self.vocab = data["vocab"]
        
        self.image_path = []
        self.text = []
        self.targets = []
        for i, (image_id, text, target) in enumerate(zip(data["images"]["img_ids"], data["images"]["texts"], data["images"]["targets"])):
            
            targets_name = data["targets_names"][str(target)]
            if targets_name != "Dudoso":
                path_image = f"./{targets_name}{os.sep}img_{str(image_id).zfill(7)}.jpg"
                if os.path.exists(path_image):
                    self.image_path.append(path_image)
                    self.text.append(text)
                    if target == 4:
                        self.targets.append(0)
                    else:
                        self.targets.append(target - 1)
                
                        
                        
                        
                else: 
                    pass
        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        
        image_path = self.image_path[index]
        text = self.text[index]
        label = self.targets[index]

        image = Image.open(image_path).convert("RGB")
        #image = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
        
        if self.transform is not None:
            convert = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            image = convert(image)
            
        return image, label
    
    
def load_split_data(datadir, valid_size = .2):
    
    train_transforms = transforms.Compose([#transforms.RandomRotation(30),  # data augmentations are great
                                       #transforms.RandomResizedCrop(224),  # but not in this case of map tiles
                                       transforms.RandomHorizontalFlip(),
                                       transforms.Resize((224,224)),
                                       #transforms.Pad((3, 10, 30, 50)),
                                       transforms.ToTensor(),
                                       #transforms.Normalize([0.485, 0.456, 0.406], # PyTorch recommends these but in this
                                    #                    [0.229, 0.224, 0.225]) # case I didn't get good results
                                       ])

    model_dataset = ImageTextData(datadir, transform=train_transforms)
    total_lenght = len(model_dataset)
    test_lenght = int(total_lenght * .2)   
    train_lenght = total_lenght - test_lenght
    
    train_data, test_data = torch.utils.data.random_split(model_dataset, [train_lenght, test_lenght])  

    print(f'Calculando pesos de train, tamaño del dataset {len(train_data)}')
    weights_train = make_weights_for_balanced_classes(train_data.dataset.targets, train_data.indices, 2)  
    weights_train = torch.DoubleTensor(weights_train)      
    sampler_train = torch.utils.data.sampler.WeightedRandomSampler(weights_train, len(weights_train))   
    
    print(f'Calculando pesos de test, tamaño del dataset {len(test_data)} ')
    weights_test = make_weights_for_balanced_classes(train_data.dataset.targets, test_data.indices, 2)    
    weights_test = torch.DoubleTensor(weights_test)      
    sampler_test = torch.utils.data.sampler.WeightedRandomSampler(weights_test, len(weights_test))  
    
    print("Creados los DataLoader para train y test")
    trainloader = torch.utils.data.DataLoader(train_data, sampler=sampler_train, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, sampler=sampler_test, batch_size=64)
    return trainloader, testloader