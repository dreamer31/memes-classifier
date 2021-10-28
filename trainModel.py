import torch, PIL, os


class Train():
    
    def __init__(self, model, optimizer, criterion, train_loader, test_loader, epochs = 5, device = 'cuda') -> None:
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device
        self.model.to(self.device)
        
        self.train_losses = []
        self.test_losses = []
        
    def get_model(self):
        return self.model
    
    def get_info(self):
        return self.train_losses, self.test_losses

    def train_model(self) -> None:

        self.resumen_train()
        steps = 0
        running_loss = 0
        print_every = 10
            
        for epoch in range(self.epochs):
            try :
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
                    
                    if steps % print_every == 0:
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
                                total += len(labels)

                        self.train_losses.append(running_loss/len(self.train_loader))
                        self.test_losses.append(test_loss/len(self.test_loader))  
                                          
                        print(f"Epoch {epoch+1}/{self.epochs}.. "
                            f"Train loss: {running_loss/print_every:.3f}.. "
                            f"Test loss: {test_loss/len(self.test_loader):.3f}.. "
                            f"Test accuracy: {float(accuracy)/float(total):.3f}")
                        
                        running_loss = 0
                        accuracy = 0
                        total =0
                        self.model.train()
                        
            except PIL.UnidentifiedImageError as error:
                print(error)
                er = str(error).split("'")
                os.remove(er[1])
                self.train_model()
                
                
                
    def resumen_train(self):
        
        print("==== Iniciando entrenamiento ==== \n")
        print(f'''Los parametros a usar son
              model: {self.model}
              optimizer: {self.optimizer}
              criterion: {self.criterion}
              epocas: {self.epochs}
              ''')
        
        