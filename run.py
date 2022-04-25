from src.classifier import process_data_bert
from src.model import CNN,  ModelMixBert, BertModelClassification
from transformers import BertModel  

import sys
import torch

from transformers import logging
logging.set_verbosity_error()


def load_model(model_path):
    """
    Load a model from a path
    
    :param model_path: path of the model
    
    :return: model
    
    """
    model = torch.load(model_path)
    return model



def load_bert():
    """
    Load Bert Model
    
    :param model_path: path of the model
    
    :return: model
    
    """
    PATH = './weight_models/bert_cnn'
    model_text = BertModel.from_pretrained('bert-base-uncased')
    model_text = BertModelClassification(model_text, 256)
    model_image = CNN(256)
    model = ModelMixBert(model_image, model_text, 512, 3)
    model.load_state_dict(torch.load(PATH))
    return model

def predict(model, move, classes, show_info):

    """
    Predict the images in the directory
    
    :param model: model
    
    :return: list of tensors of the images with text
    
    """
    model.eval()
    process_data_bert(model, './img_class', classes, move=move, show_info=show_info)

if __name__ == "__main__":
    
    model_name = sys.argv[1]
    mode_classifier = sys.argv[2]
    move_image = True if sys.argv[3] == "True" else False
    show_info = bool(sys.argv[4])
    classes = ('Meme', 'No Meme', 'Sticker')

    print("Cargando modelo..")
    if model_name == "bert":
        model = load_bert()

    print("Prediciendo...")
    predict(model, move_image, classes, show_info)

        
