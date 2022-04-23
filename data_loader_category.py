from cgi import test
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from utils import weights_balanced, make_weights_for_balanced_classes
from tokenizer_category import TokernizerMemeCategory
from torch.nn.utils.rnn import pad_sequence
from category import categories
from transformers import BertTokenizer, PreTrainedTokenizerFast, AutoTokenizer
from data_augmentation import DataAugmentator



import torch
import pandas as pd



class DataLoaderCategory(Dataset):
    def __init__(self, data_path, shuffle=True, num_workers=4, data_augmentation = False, BERT=False):
        self.data_path = data_path
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.data_augmentation = data_augmentation
        self.BERT = BERT
        self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

        self.transforms = transforms.Compose([transforms.Resize((56, 56)),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.augmentator = DataAugmentator()
        self.tokenizer = TokernizerMemeCategory()
        
        self.data = pd.read_csv(self.data_path)
        self.label = []
        self.tematica = list(self.data["TEMATICA"])
        self.image_links = list(self.data['links'])
        self.text = []
        self.text_bert = []
        self.mask_bert = []
        self.image = []
        self.tematicas_name = ()
        self.cont_tematica = {}

        cont = 0
        aux_tematica = {}
        for tematica in self.tematica:
            if tematica not in aux_tematica:
                aux_tematica[tematica] = cont
                self.tematicas_name += (categories[tematica],)
                self.cont_tematica[categories[tematica]] = 0
                cont+=1

        image_cont = 0
        for image_link in self.image_links:
            if image_link != "No":
                image_link = "./categoria/images/" + image_link
                image = self.process_image(image_link)
                text = self.data["text_manual"][image_cont]
                self.label.append(aux_tematica[self.tematica[image_cont]])
                self.cont_tematica[categories[self.tematica[image_cont]]] += 1
                text_process = self.process_text(text)

                if self.BERT:
                    self.process_text_bert(self.data["text_manual"][image_cont])

                if self.data_augmentation:

                    self.process_augmentation(image_link, text,  aux_tematica[self.tematica[image_cont]])
                    self.cont_tematica[categories[self.tematica[image_cont]]] += 1

            image_cont += 1
        self.text = pad_sequence(self.text,
                                         batch_first=True,
                                         padding_value=self.tokenizer.get_vocab()["<pad>"])

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index: int) -> tuple:
        
        image = self.image[index]
        text = self.text[index]
        label = self.label[index]

        if self.BERT:
            text_bert = self.text_bert[index]
            mask_bert = self.mask_bert[index]
            return image, text, text_bert, mask_bert, label

        
        return image, text, label


    def get_tokenizer(self) -> TokernizerMemeCategory:
        """
        Return tokenizer
        """
        return self.tokenizer

    def get_vocab(self) -> dict:
        """
        Return vocab
        """
        return self.tokenizer.get_vocab()

    
    def get_len_vocab(self) -> int:
        """
        Return len vocab
        """
        return len(self.tokenizer.get_vocab())

    def get_tematicas_name(self):
        return self.tematicas_name


    def get_cont_tematica(self):
        return self.cont_tematica


    def process_text_bert(self, text: str) -> torch.Tensor:
        """
        Process text and return tensor
        """
        encoded = self.bert_tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            max_length = 32,
            pad_to_max_length = True,             
            return_attention_mask = True,
            return_tensors='pt',
        )

            
        self.text_bert.append(encoded['input_ids'].flatten())
        self.mask_bert.append(encoded['attention_mask'].flatten())


    
    def process_image(self, image_path: str) -> torch.Tensor:
        """
        Process image and return tensor
        """
        
        image = Image.open(image_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)


        self.image.append(image)
        return image

    def process_text(self, text: str) -> torch.Tensor:
        """
        Process text and return tensor
        """
        text = self.tokenizer.tokenize(text)
        self.text.append(torch.tensor(text))
        return torch.tensor(text)

    def process_augmentation(self, image, text, label):
        """
        Data augmentation
        """
        image_augment = self.augmentator.augment_image(image)
        text_augment = self.augmentator.augment_text(text)
        self.process_text_bert(text_augment)
        self.image.append(image_augment)
        self.label.append(label)
        self.text.append(self.process_text(text_augment))
        return image


def load_split_data(datadir: str, test_size: float = 0.2, batch_size: int = 32, data_augmentation: bool = False, BERT = False):  
    
    model_dataset = DataLoaderCategory(datadir, data_augmentation=data_augmentation, BERT=BERT)
    print("cantidad datos", len(model_dataset))

    total_lenght = len(model_dataset)
    test_lenght = int(total_lenght * test_size)
    train_lenght = total_lenght - test_lenght
    

    train_data, test_data = torch.utils.data.random_split(model_dataset,
                                                         [train_lenght, test_lenght])

    weights_train = make_weights_for_balanced_classes(train_data.dataset.label, train_data.indices, 17)
    weights_train = torch.DoubleTensor(weights_train)


    sampler_train = torch.utils.data.sampler.WeightedRandomSampler(weights_train, len(weights_train))

    train_loader = torch.utils.data.DataLoader(train_data, sampler=sampler_train , batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader, model_dataset


def generate_batch(batch):
    """
    Generate batch of data
    """

    label_list, text_list, image_list = [], [], []
    for (image, text, label) in batch:
        image_list.append(image)
        text_list.append(text)
        label_list.append(label)
    
    return torch.stack(image_list), text_list, torch.tensor(label_list)