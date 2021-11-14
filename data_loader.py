from torchvision import transforms
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from utils import make_weights_for_balanced_classes

import os
import torch

torch.manual_seed(42)


class ImageTextData(Dataset):
    
    """
    Custom Dataset for images and text of meme dataset
    
    """

    def __init__(self, df: str, transform: bool=False, only_meme: bool=False, imagen_out: bool=True):

        # Dict with the initial info
        self.df = df

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
                        self.targets.append(
                            0) if only_meme else self.targets.append(2)
                    else:
                        self.targets.append(target - 1)

                    text_tensor = torch.tensor(text)
                    self.text.append(text_tensor)

                else:
                    pass

        self.text_padding = pad_sequence(self.text,
                                         batch_first=True,
                                         padding_value=self.vocab["<pad>"])

    def __len__(self) -> int:
        
        """
        Get length of dataset
        
        return: int -> length of dataset
        """
        
        return len(self.image_path)

    def __getitem__(self, index: int) -> tuple:
        
        """
        Get item from dataset with index
        Its for iterator
        
        :param index: int -> index of item
        
        return: tuple -> (image, text, target)
        
        """

        image = self.image_path[index]
        label = self.targets[index]
        text = self.text[index]
        
        return image, text, label
    
    def get_transform(self) -> transforms.Compose:
        return self.transform

def load_split_data(datadir: str, batch_size: int=64, test_size: float=.2, imagen_out: bool=True) -> tuple:
    
    """
    Create the split dataset for train an test with test_size
    Balanced classes for underfitting and overfitting
    
    :param datadir: str -> path to data
    :param bath_size: int -> batch size
    :param test_size: float -> test size
    :param imagen_out: bool -> if true, return image, else text
    
    return: tuple -> (train_data, test_data)
    
    """

    model_dataset = ImageTextData(datadir,
                                  imagen_out=imagen_out,
                                  )

    total_lenght = len(model_dataset)
    test_lenght = int(total_lenght * test_size)
    train_lenght = total_lenght - test_lenght

    train_data, test_data = torch.utils.data.random_split(model_dataset,
                                                          [train_lenght, test_lenght])

    weights_train = make_weights_for_balanced_classes(train_data.dataset.targets,
                                                      train_data.indices,
                                                      3)
    weights_train = torch.DoubleTensor(weights_train)
    sampler_train = torch.utils.data.sampler.WeightedRandomSampler(
        weights_train, len(weights_train))

    weights_test = make_weights_for_balanced_classes(
        train_data.dataset.targets, test_data.indices, 3)
    weights_test = torch.DoubleTensor(weights_test)
    sampler_test = torch.utils.data.sampler.WeightedRandomSampler(
        weights_test, len(weights_test))

    trainloader = torch.utils.data.DataLoader(train_data,
                                              sampler=sampler_train,
                                              batch_size=batch_size,
                                              collate_fn=generate_batch
                                              )
    testloader = torch.utils.data.DataLoader(test_data,
                                             sampler=sampler_test,
                                             batch_size=batch_size,
                                             collate_fn=generate_batch
                                             )
    return trainloader, testloader


def generate_batch(batch: tuple) -> tuple:
    
    """
    Function for generate batch for data loader
    Load and transform image
    Creade pad_sequence for text
    
    :param batch: tuple -> (image, text, target)
    
    return: tuple -> (image, text, target)
    
    """

    train_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    label_list, text_list, image_list = [], [], []
    for (_image, _text, _label) in batch:
        label_list.append(_label)
        text_list.append(_text)

        image = Image.open(_image).convert("RGB")
        if train_transforms is not None:
            image = train_transforms(image)
        image_list.append(image)

    return (
        torch.stack(image_list),
        pad_sequence(text_list, batch_first=True, padding_value=1),
        torch.tensor(label_list)
    )
    