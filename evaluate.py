import torch
import matplotlib.pyplot as plt
from model import FCNN
from data_processing import load_test_data
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Učitavanje preprocessora
preprocessor = joblib.load('preprocessor.joblib')

# Učitavanje test podataka
X_test = load_test_data(preprocessor)

# Provera dimenzija ulaznih podataka za testiranje
print(f"Dimenzije ulaznih podataka za testiranje: {X_test.shape[1]}")

# Inicijalizacija modela i učitavanje treniranih težina
input_dim = X_test.shape[1]
model = FCNN(input_dim)
model.load_state_dict(torch.load('fcnn_model.pth')) # Učitava trenirane težine modela sa diska.
print("Model učitan iz 'fcnn_model.pth'")

# Funkcija za predikciju
def predict(model, test_data):
    model.eval() # Postavlja model u režim evaluacije (deaktivira Dropout)
    with torch.no_grad(): #  Isključuje računanje gradijenata radi uštede memorije i ubrzavanja.
        test_tensor = torch.tensor(test_data, dtype=torch.float32) # Konvertuje test podatke u PyTorch tensor.
        predictions = model(test_tensor) #  Prolazi test podatke kroz model da bi dobio predikcije.
        return torch.softmax(predictions, dim=1).numpy() # Primjenjuje softmax funkciju na predikcije kako bi dobili vjerovatnoce i konvertuje ih u NumPy niz.

# Predikcije na test podacima
predictions = predict(model, X_test)
print(predictions)

# Učitavanje tačnosti iz treninga
train_accuracy = joblib.load('train_accuracy.joblib')  # Učitaj podatke o tačnosti iz treninga
val_accuracy = joblib.load('val_accuracy.joblib')      # Učitaj podatke o tačnosti iz validacije

# Prikazivanje grafikona tačnosti
epochs = range(1, len(train_accuracy) + 1)  # Koristi stvarni broj epoha
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracy, label='Training Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Funkcija za odredjivanje klase sa najvecom vjerovatnocom
def get_predicted_classes(predictions):
    return np.argmax(predictions, axis=1)

def plot_prediction_distribution(predictions):
    plt.figure(figsize=(12, 6))
    for i in range(3):  # Tri klase
        plt.hist(predictions[:, i], bins=50, alpha=0.5, label=f'Class {i}')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of Predicted Probabilities for Each Class')
    plt.show()

# Predikcije na test podacima
predictions = predict(model, X_test)

# Prikazivanje distribucije verovatnoća
plot_prediction_distribution(predictions)

predicted_classes = get_predicted_classes(predictions)

# Ispis prvih nekoliko predikcija
print(predicted_classes[:50])
