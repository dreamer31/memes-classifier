import torch
import easyocr

from PIL import Image
from torchvision import transforms
from pathlib import Path

import os


def get_files_from_directory(path):
    
    """"
    Get all paths of files in a directory
    
    :param path: path of the directory
    
    :return: iterator of paths
    
    """
    
    file_path = Path(path)
    iterator = file_path.iterdir()
    return iterator


def recognize_text(img_path, reader):
    
    """
    Recognize text from an image
    
    :param img_path: path of the image
    :reader: reader of easyocr
    
    :return: list of recognized text
    
    """
    # try:
    text = reader.readtext(img_path)
    return text
    # except ValueError as e:
    #     print(f"Eliminada {img_path}")
    #     os.remove(img_path)
  


def image_to_text(image_path, vocab, reader):
    
    """
    Transform a image to text
    
    :param image_path: path of the image
    :param vocab: vocabulary of the text
    :param reader: reader of easyocr
    
    :return: tensor of the text
    
    """
    
    text_predict = recognize_text(image_path, reader)
    text_tensor = [1]
    
    for element in text_predict:
        for word in element[1].split():
            word = word.lower()
            if word in vocab:
                text_tensor.append(vocab[word])
        
    text_tensor = torch.tensor([text_tensor])
    text_tensor = text_tensor.type(torch.int64)
    return text_tensor

def load_image(image_path):
    
    """
    Load a image from a path
    
    :param image_path: path of the image
    
    :return: Tensor of the image
    
    """
    
    train_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    image = Image.open(image_path).convert("RGB")    
    image = train_transforms(image)
    image = [image]
    image = torch.stack(image)
    return image

def process_data(vocab, model, init_directory):
    
    """
    Process all images in a directory
    
    :param vocab: vocabulary of the text
    
    :return: list of tensors of the images with text
    
    """
    
    results = []
    reader = easyocr.Reader(['es'])
    iter_image = get_files_from_directory(init_directory)
    for image in iter_image:
        #print(str(image))
        text_tensor = image_to_text(str(image), vocab, reader)
        image_loaded = load_image(str(image))
        
        predict = model.forward(image_loaded, text_tensor)
        val, ind = predict.squeeze(1).max(1)
        move_image(str(image), ind.item())
        
    return results
    
import shutil    
    
def move_image(path, classify):
    
    
    #print(f"La imagen {path} se clasifico como {classify}")
    
    meme_path = "./meme-class"
    no_meme_path = "./no-meme-class"
    sticker_path = "./sticker-class"
    
    if classify == 0:
        shutil.move(path, meme_path)
    elif classify == 1:
        shutil.move(path, no_meme_path)
    else:
        shutil.move(path, sticker_path)
    
    
    