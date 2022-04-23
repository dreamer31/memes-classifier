import nlpaug.augmenter.word as naw
from torchvision import transforms
from PIL import Image
from deep_translator import GoogleTranslator




class DataAugmentator:

    def __init__(self):

        self.synonym_augmentator = naw.SynonymAug(aug_src='wordnet', lang='spa')
        self.random_augmentator = naw.RandomWordAug(action='delete', name='RandomWord_Aug', aug_min=1, aug_max=10, aug_p=0.3, stopwords=None,
                                                    target_words=None, tokenizer=None, reverse_tokenizer=None, stopwords_regex=None, verbose=0)

        self.flip = transforms.Compose([transforms.Resize((56, 56)),
                                        transforms.RandomVerticalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.crop = transforms.Compose([transforms.Resize((56, 56)),
                                        transforms.RandomCrop(56),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


    def synonymAugmentator(self, text):
        """
        Synonym Augmentator
        """
        
        return self.synonym_augmentator.augment(text)


    def randomAugmentator(self, text):
        """
        Random Augmentator
        """
        return self.random_augmentator.augment(text)

    def back_translation_text(self, text):
        """
        Back translation text
        """
        translator_en = GoogleTranslator(source='es', target='en')
        translator_es = GoogleTranslator(source='en', target='es')
        text_en = translator_en.translate(text)
        return translator_es.translate(text_en)

    def augment_text(self, text):
        """
        Augment text
        """
        return self.back_translation_text(text)

    def augment_image(self, image):
        """
        Augment image
        """
        image = Image.open(image).convert("RGB")
        return self.flip(image)

    