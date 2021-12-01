import torch
import PIL
import os
import hiddenlayer as hl
import sys

from utils import confusion_matrix_plot, plot_images, plot_confusion_matrix
from sklearn.metrics import confusion_matrix

torch.backends.cudnn.deterministic = True


class Train():
    def __init__(self, model, optimizer, criterion, train_loader, test_loader,
                 epochs=100, prints_every=1, device='cuda', writer=None, show_matrix=False, show_image=False) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.prints_every = prints_every
        self.device = device
        self.writer = writer
        self.model.to(self.device)
        self.show_matrix = show_matrix
        self.show_image = show_image

        self.train_losses = []
        self.test_losses = []

        self.y_true = []
        self.y_predicted = []
        self.images_show = []

        self.classes = ('Meme', 'No Meme', 'Sticker')

    def get_model(self):
        
        """
        Return model object
        
        :return: model
        
        """
        
        return self.model

    def get_info(self) -> tuple:
        
        """
        Return information about the model
        
        :return: model_info
        """
        
        return self.train_losses, self.test_losses

    def train_model(self) -> None:
        
        """
        function for training the model
        
        """

        self.resumen_train()
        steps = 0
        running_loss = 0
        
        for epoch in range(self.epochs):
            
            try:
                for image, text, labels in self.train_loader:
                    steps += 1
                
                    image = image.to(self.device)
                    labels = labels.to(self.device)
                    text = text.type(torch.int64).to(self.device)
                    
                    self.optimizer.zero_grad()
                    predict = self.model.forward(image, text)

                    loss = self.criterion(predict, labels)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()
                    
                    if steps % self.prints_every == 0:
                        test_loss = 0
                        accuracy = 0
                        total = 0
                        self.model.eval()

                        with torch.no_grad():
                            for image, text, labels in self.test_loader:

                                image = image.to(self.device)
                                text = text.type(torch.int64).to(self.device)
                                labels = labels.to(self.device)
                                
                                predict = self.model.forward(image, text)
                                batch_loss = self.criterion(predict, labels)
                                test_loss += batch_loss.item()

                                val, ind = predict.squeeze(1).max(1)
                                accuracy += (ind == labels).sum()

                                self.y_true.extend(labels.cpu().numpy())
                                self.y_predicted.extend(ind.cpu().numpy())
                                self.images_show.extend(image.cpu().numpy())

                                total += len(labels)

                        self.train_losses.append(running_loss/len(self.train_loader))
                        self.test_losses.append(test_loss/len(self.test_loader))
                        
                        if self.writer is not None:
                            self.writer.add_scalar("Loss/test", 
                                                   running_loss/self.prints_every, steps)  
                            self.writer.add_scalar("Acc/test", 
                                                   float(accuracy)/float(total), steps) 
                            
                        if self.show_matrix:
                            con = confusion_matrix(self.y_true, self.y_predicted)      
                            plot_confusion_matrix(con, self.classes)                 
                            
                            
                        if self.show_image:
                            plot_images(self.images_show, self.y_true, self.y_predicted, self.classes)
                
                        sys.stdout.write(f"\rEpoch {epoch+1}/{self.epochs}.. "
                            f"Train loss: {running_loss/self.prints_every:.3f}.. "
                            f"Test loss: {test_loss/len(self.test_loader):.3f}.. "
                            f"Test accuracy: {float(accuracy)/float(total):.3f}")
                        
                        running_loss = 0
                        accuracy = 0
                        total = 0
  
                        self.y_true = []
                        self.y_predicted = []
                        self.images_show = []
             
                        self.model.train()
          
            except PIL.UnidentifiedImageError as error:
                print(error)
                er = str(error).split("'")
                os.remove(er[1])
                self.train_model()

    def resumen_train(self) -> None:
        
        """
        Print information about the model and train
        """

        print("==== Iniciando entrenamiento ==== \n")
        print(f'''Los parametros a usar son
              model: {self.model}
              optimizer: {self.optimizer}
              criterion: {self.criterion}
              epocas: {self.epochs}
              ''')

    def save_model(self, path: str) -> None:
        
        """
        Save the model in a file
        
        :param path: path to save the model
        
        :return: None
        
        """
        
        torch.save(self.model.state_dict(), path)

    def create_graph(self, image, text) -> None:
        
        """
        Create a graph with information about the model
        
        :param image: image to predict
        :param text: text to predict
        
        """
        
        hl.build_graph(model=self.model,
                       args=(image.to(self.device),
                             text.type(torch.int64).to(self.device)))