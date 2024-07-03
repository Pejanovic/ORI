import torch
import torch.nn as nn

class FCNN(nn.Module): #NN je osnovna klasa za neuronske mreze i pruza osnovne funkcionalnosti npr. definisanje slojeva i njihovo povezivanje
    def __init__(self, input_dim):
        super(FCNN, self).__init__() #  konstruktor roditeljske klase sa svim meotadama i atributima
        self.fc1 = nn.Linear(input_dim, 128) # Prvi sloj je fully connected sloj koji uzima input dim broja ulaznih jedinica i ima 128 izlaznih jedinica
        self.fc2 = nn.Linear(128, 64) # Drugi sloj je fully connected sloj koji uzima 128 broja ulaznih jedinica i ima 64 izlaznih jedinica...
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 3)
        self.dropout = nn.Dropout(p=0.3) # Definisemo dropout sloj sa vjerovatnocom izbacivanja 0.4. Koristimo ga da smanjimo overfitting tako sto nasumicno postavljamo odredjeni broj 1 na 0

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # Nakon linearne transformacije, izlaz prolazi kroz ReLU (Rectified Linear Unit) aktivacionu funkciju. Ulazni podaci x prolaze kroz prvi potpuno povezani sloj (fc1)
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
