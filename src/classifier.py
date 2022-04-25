import torch
import easyocr


from src.tokenizers.tokenizer import TokenizerMeme
from PIL import Image
from torchvision import transforms
from pathlib import Path
from transformers import BertTokenizer, PreTrainedTokenizerFast, AutoTokenizer


import numpy as np
import cv2
import tempfile


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
    #image = Image.open(img_path)
    #image = image.convert('RGB')
    #image = np.array(image)
    #im= cv2.bilateralFilter(image,5, 55,60)
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #_, im = cv2.threshold(im, 240, 255, 1)
    
    
    text = reader.readtext(img_path)
    return text


def image_to_text(image_path, vocab, reader, tokenizer):
    
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
            word = tokenizer.clean_text(word.lower())
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
    
    train_transforms = transforms.Compose([transforms.Resize((56, 56)),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    image = Image.open(image_path).convert("RGB")    
    image = train_transforms(image)
    image = [image]
    image = torch.stack(image)
    return image

def process_data(vocab, model, init_directory, move=False):
    
    """
    Process all images in a directory
    
    :param vocab: vocabulary of the text
    
    :return: list of tensors of the images with text
    
    """
    
    results = []
    tokenizer = TokenizerMeme(vocab)
    reader = easyocr.Reader(['en'])
    iter_image = get_files_from_directory(init_directory)
    for image in iter_image:
        text_tensor = image_to_text(str(image), vocab, reader, tokenizer)
        image_loaded = load_image(str(image))
        predict = model.forward(image_loaded, text_tensor)
        val, ind = predict.squeeze(1).max(1)
        results.append((str(image), ind.item()))
        if move:
            move_image(str(image), ind.item())
        
    return results

def image_to_text_bert(image_path, reader):
    
    """
    Transform a image to text
    
    :param image_path: path of the image
    :param vocab: vocabulary of the text
    :param reader: reader of easyocr
    
    :return: tensor of the text
    
    """
    
    text_predict = recognize_text(image_path, reader)
    text = ""
    
    for element in text_predict:

        for word in element[1].split():
            word = word.lower()
            text += word + " "
            

    return text

def process_data_bert(model, init_directory, move=False):
    
    """
    Process all images in a directory
    
    :param vocab: vocabulary of the text
    
    :return: list of tensors of the images with text
    
    """
    
    results = []
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    reader = easyocr.Reader(['en'])
    iter_image = get_files_from_directory(init_directory)

    for image in iter_image:
        text_image = image_to_text_bert(str(image), reader)
        encoded = bert_tokenizer.encode_plus(
            text=text_image,
            add_special_tokens=True,
            max_length = 16,
            pad_to_max_length = True,            
            return_attention_mask = True,
            return_tensors='pt',
        )
        mask = encoded['attention_mask'].flatten().unsqueeze(0)
        text_tensor = encoded['input_ids'].flatten().unsqueeze(0)

        image_loaded = load_image(str(image))

        predict = model.forward(image_loaded, text_tensor, mask)

        val, ind = predict.squeeze(1).max(1)
        results.append((str(image), ind.item()))
        if move:
            move_image(str(image), ind.item())
        
    return results


    
import shutil    
    
def move_image(path, classify):
        
    meme_path = "./meme-class"
    no_meme_path = "./no-meme-class"
    sticker_path = "./sticker-class"
    
    if classify == 0:
        shutil.move(path, meme_path)
    elif classify == 1:
        shutil.move(path, no_meme_path)
    else:
        shutil.move(path, sticker_path)
    
    

def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False,   suffix='.png')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(300, 300))
    return temp_filename