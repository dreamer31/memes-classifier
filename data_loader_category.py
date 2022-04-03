from cgi import test
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


import torch
import pandas as pd



class DataLoaderCategory(Dataset):
    def __init__(self, data_path, shuffle=True, num_workers=4):
        self.data_path = data_path
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.transforms = transforms.Compose([transforms.Resize((32, 32)),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        
        self.data = pd.read_csv(self.data_path)
        self.label = []
        self.tematica = list(self.data["TEMATICA"])
        self.image_links = list(self.data['links'])
        self.text = list(self.data['text'])
        self.image = []

        cont = 0
        aux_tematica = {}
        for tematica in self.tematica:
            if tematica not in aux_tematica:
                aux_tematica[tematica] = cont
                self.label.append(cont)
                cont+=1
            else:
                self.label.append(aux_tematica[tematica])

        
            

        unique_list = list(dict.fromkeys(self.label))
        print(unique_list)

        for image_link in self.image_links:
            if image_link != "No":
                image_link = "./categoria/images/" + image_link
                self.process_image(image_link)


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index: int) -> tuple:
        
        image = self.image[index]
        text = self.text[index]
        label = self.label[index]
        
        return image, text, label

    
    def process_image(self, image_path: str) -> torch.Tensor:
        """
        Process image and return tensor
        """
        
        image = Image.open(image_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        
        self.image.append(image)
        return image


def load_split_data(datadir: str, test_size: float = 0.2, batch_size: int = 32):  
    
    model_dataset = DataLoaderCategory(datadir)

    total_lenght = len(model_dataset)
    test_lenght = int(total_lenght * test_size)
    train_lenght = total_lenght - test_lenght

    train_data, test_data = torch.utils.data.random_split(model_dataset,
                                                         [train_lenght, test_lenght])


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader