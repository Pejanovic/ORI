import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import FCNN
import torch.nn as nn
from data_processing import load_data
import joblib

# Učitavanje i priprema podataka
X_train, X_val, y_train, y_val, preprocessor = load_data()

# Provera dimenzija ulaznih podataka za trening
print(f"Dimenzije ulaznih podataka za trening: {X_train.shape[1]}")

# Konverzija u PyTorch tenzore
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Definisanje modela, gubitne funkcije i optimizatora
input_dim = X_train.shape[1]
model = FCNN(input_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # Dodata L2 regularizacija sa weight_decay

# Funkcija za trening modela
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=40, patience=5):
    train_accuracy = []
    val_accuracy = []
    best_val_loss = float('inf') # incijalizujemo best_val_loss na beskonacno da bi u prvoj iteraciji usli u if 
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0) # akumulira ukupni gubutak za cijelu epohu. loss.item() vraca skalar iz tenzora koji predstavlja loss. 
                                                        # inputs_size predstavlja broj uzoraka u batchu(32) // inputs -> tenzor (batch_size, num_features)
            predicted = torch.argmax(outputs, dim=1) # outputs je matrica predikcija modela. torch.argmax uzima argmax duz odredjene dimenzije. Sto znaci da za svaki batch odredjuje klasu sa najvecom vjerovatnocom
                                                    # predicted je tenzor koji sadrzi predikcije klasa za svaki uzorak
            correct_train += (predicted == targets).sum().item() # predicted == targets vraca true/false - predikcije tacne/netacne // correct_train varijabla koja
                                                                # akumulira broj tacnih predikcija za trenutni batch i dodaje ih ukupnom broju za cijelu epohu
            total_train += targets.size(0) # akumulira ukupan broj uzoraka tokom epohe

        epoch_loss = running_loss / len(train_loader.dataset) # prosjecan gubitak po uzorku za cijelu epohu
        train_acc = correct_train / total_train # tacnost modela -> broj tacnih predikcija / ukupan broj uzoraka
        train_accuracy.append(train_acc)

        # Validacija modela
        model.eval() # postavlja model u rezim evaluacije -> najvise zbog dropouta jer se ponasa drugacije u rezimu evaluacije
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad(): # iksljucuje racunanje gradijenta. Cilj ustediti memoriju i vrijeme jer gradijenti nisu potrebni
            for inputs, targets in val_loader: 
                outputs = model(inputs) # outputs -> predikcije koje vraca model
                loss = criterion(outputs, targets) # -> gubitak izmedju predikcija i stvarnih vrijednosti
                val_loss += loss.item() * inputs.size(0) # -> akumulacija gubitaka za trenutni batch

                predicted = torch.argmax(outputs, dim=1) # -> predikcije za svaki uzorak u trenutnom batchu // uzimamo indeks klase sa najvecom vjerovatnocom
                correct_val += (predicted == targets).sum().item() # tacan broj predikcija
                total_val += targets.size(0) # ukupan broj uzoraka tokom epohe

        val_loss /= len(val_loader.dataset) # prosjecan gubitak po uzorku za trenutnu epohu
        val_acc = correct_val / total_val # tacnost
        val_accuracy.append(val_acc)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_wts = model.state_dict() # Čuva trenutno stanje modela (težine i pristranosti) u best_model_wts
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print('Early stopping!')
                model.load_state_dict(best_model_wts)
                break

    return train_accuracy, val_accuracy

# Treniranje modela
train_accuracy, val_accuracy = train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=40, patience=5)

# Čuvanje modela i preprocessora
torch.save(model.state_dict(), 'fcnn_model.pth')
joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(train_accuracy, 'train_accuracy.joblib')  # Čuvanje tačnosti tokom treninga
joblib.dump(val_accuracy, 'val_accuracy.joblib')      # Čuvanje tačnosti tokom validacije
print("Model sačuvan kao 'fcnn_model.pth' i preprocessor kao 'preprocessor.joblib'")
print("Tačnosti tokom treninga i validacije sačuvane kao 'train_accuracy.joblib' i 'val_accuracy.joblib'")
