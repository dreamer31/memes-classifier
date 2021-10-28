from torchtext.vocab import vocab
from torchvision import transforms
from torchtext.legacy import data
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from BalancedClass import make_weights_for_balanced_classes

import os
import torch


class ImageTextData(Dataset):
    
    def __init__(self, df, transform = False, only_meme = False, imagen_out = True):
        
        # Dict with the initial info
        self.df = df
        
        # Transform for image data
        self.transform = transform
        
        # Vocab for text data
        self.vocab = df["vocab"]
        self.vocab["<pad>"] = 1
        
        self.image_out = imagen_out   
                        
        # Data iterator
        self.image_path = []
        self.text = []
        self.targets = []
        
        # Process data
        for i, (image_id, text, target) in enumerate(zip(df["images"]["img_ids"],
                                                         df["images"]["texts"],
                                                         df["images"]["targets"])):
            
            targets_name = df["targets_names"][str(target)]
            if targets_name != "Dudoso":
                path_image = f"./{targets_name}{os.sep}img_{str(image_id).zfill(7)}.jpg"
                
                if os.path.exists(path_image):
                    self.image_path.append(path_image)
                    if target == 4:
                        self.targets.append(0) if only_meme else self.targets.append(2)
                    else:
                        self.targets.append(target - 1)
                        
                    # phrase = ''
                    # keys_vocab = list(self.vocab.keys())
                    # for word in text:
                    #     phrase += keys_vocab[word] + ' '
                    
                    text_tensor = torch.tensor(text)
                    self.text.append(text_tensor)
                    #self.text.append(phrase)
                    
                else: 
                    pass
                
        self.text_padding = pad_sequence(self.text, batch_first=True, padding_value=self.vocab["<pad>"])        
        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, index):
        
        image = self.image_path[index]
        label = self.targets[index]
        text = self.text_padding[index]
                
        if self.image_out:
            image = Image.open(image).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, text, label
    
    
def load_split_data(datadir, batch_size = 64, valid_size = .2, imagen_out = True):
    
    train_transforms = transforms.Compose([transforms.Resize((30, 30)), 
                                           transforms.ToTensor(), 
                                           transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    model_dataset = ImageTextData(datadir, transform=train_transforms, imagen_out=imagen_out)
    total_lenght = len(model_dataset)
    test_lenght = int(total_lenght * .2)   
    train_lenght = total_lenght - test_lenght
    
    train_data, test_data = torch.utils.data.random_split(model_dataset, [train_lenght, test_lenght])  

    weights_train = make_weights_for_balanced_classes(train_data.dataset.targets, train_data.indices, 3)  
    weights_train = torch.DoubleTensor(weights_train)      
    sampler_train = torch.utils.data.sampler.WeightedRandomSampler(weights_train, len(weights_train))   
    
    weights_test = make_weights_for_balanced_classes(train_data.dataset.targets, test_data.indices, 3)    
    weights_test = torch.DoubleTensor(weights_test)      
    sampler_test = torch.utils.data.sampler.WeightedRandomSampler(weights_test, len(weights_test))  
    
    trainloader = torch.utils.data.DataLoader(train_data, sampler=sampler_train, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data, sampler=sampler_test, batch_size=batch_size)
    return trainloader, testloader