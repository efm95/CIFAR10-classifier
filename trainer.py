import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import model 


class trainer:
    def __init__(self,
               net = model.CNN(),
               lr = 0.001,
               momentum=0.9,
               model_name='cnn_model.pt'):
        
        self.lr = lr
        self.momentum = momentum
        self.model_name = model_name
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), 
                                   lr=self.lr, 
                                   momentum=self.momentum)
    
    def fit(self,
            train_set,
            validation_set,
            epochs=10):
        
        train_loss = []
        val_loss = []
        
        train_acc = []
        val_acc = []
        
        for epoch in range(epochs):
            
            running_loss = 0.0
            running_total = 0
            
            running_correct = 0
            run_step = 0
            
            self.net.train()
            for i,batch in enumerate(train_set,0):
                
                inputs,labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                out = self.net(inputs)
                loss = self.loss_fn(out,labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                running_total += labels.size(0)
                with torch.no_grad():
                    _, predicted = out.max(1)
                
                running_correct += (predicted==labels).sum().item()
                run_step +=1
                
                if i % 100==0:
                    print(f'Epoch: {epoch+1} | Batch: {i} | Loss: {running_loss / run_step} | Training accuracy: {100 * running_correct / running_total}%')
            
            if validation_set != None:
                correct = 0
                total = 0
                val_loss_running = 0.0
                
                self.net.eval()
                with torch.no_grad():
                    for val in validation_set:
                        inputs_val, labels_val = val
                        inputs_val = inputs_val.to(self.device)
                        labels_val = labels_val.to(self.device)
                        
                        valid_outputs = self.net(inputs_val)
                        valid_loss = self.loss_fn(valid_outputs,labels_val)

                        val_loss_running += valid_loss.item()
                        _,predicted = valid_outputs.max(1)
                        
                        total += labels_val.size(0)
                        correct += (predicted == labels_val).sum().item()
                       
                
            val_loss.append(val_loss_running/len(validation_set))
            train_loss.append(running_loss / run_step)

            val_acc.append(100 * correct / total)
            train_acc.append(100 * running_correct / running_total)
            
            print(f'Epoch: {epoch+1} | Loss {running_loss / run_step} | Train accuracy: {train_acc[-1]}% | Validation accuracy {val_acc[-1]}%')
        
        print('SAVING MODEL')
        torch.save({'epoch': epoch,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.net.state_dict(),
                    'train_loss': running_loss,
                    }, self.model_name)

        print('Finished training')
        return train_loss, val_loss ,val_acc, train_acc
    

    def prediction(self,
                   test_set):
        with torch.no_grad():
            correct = 0
            total = 0
            self.net.eval()
            for test in test_set:
                inputs, labels = test
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data,1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            print('Accuracy of the network on 10000 test images: %d %%' %(100*correct/total))