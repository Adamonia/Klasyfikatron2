

import xlsxwriter
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from main import train_classifier, load_data, build_classifier_model, make_predictions, load_from_pb
from datetime import datetime



# sciezka_plik_in = "bbb"
# sciezka_plik_out = "bbb"

sciezka_plikow_train = None
sciezka_plik_test = None

model = None

timer_id = None
class_names = None


def browse_train_files():
    # folder z plikami treningowymi podzielonymi na klasy w osobnych folderach. Nazwy podfolderow zostana uzyte jako klasy
    filenames = filedialog.askdirectory(mustexist=True)

    if not filenames:
        return

    global sciezka_plikow_train
    sciezka_plikow_train = filenames

    label1.configure(text=f"{len(filenames)} files opened", fg="red")


def browse_test_files():
    # folder z plikami testowymi podzielonymi na klasy w osobnych folderach. Nazwy podfolderow zostana uzyte jako klasy
    filename = filedialog.askdirectory(mustexist=True)

    if not filename:
        return

    global sciezka_plik_test
    sciezka_plik_test = filename

    label1.configure(text="File Opened: " + filename)


def make_prediction_for_given_files():
    global model
    global class_names

    if not model:
        label1.configure(text=f"Brak modelu")
        return None

    # pliki
    filenames = filedialog.askopenfilenames(initialdir="/", title="Select a File",
                                            filetypes=(("Text files", "*.txt*"), ("all files", "*.*")))


    #otwieranie plikow
    data = []
    for filename in filenames:
        fileObject = open(filename, "r")
        data.append(fileObject.read())

    # uzycie modelu
    _, predicted_classes, scores = make_predictions(model, data, class_names)

    save_the_predictions(filenames, predicted_classes, scores)

    if len(filenames) < 3:
        label1.configure(text=f"{predicted_classes} is/are the predicted class(es) with {scores} % score(s).")

def save_the_predictions(filenames, predicted_classes, scores):
    # Saving the report

    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")

    workbook = xlsxwriter.Workbook('predictions_{}.xlsx'.format(dt_string))
    worksheet = workbook.add_worksheet()

    worksheet.write('A1', 'File')
    worksheet.write('B1', 'predicted class')
    worksheet.write('C1', 'score')

    for filename, predicted_class, score, row in zip(filenames, predicted_classes, scores, range(1, len(scores) + 1)):
        worksheet.write(row, 0, filename)
        worksheet.write(row, 1, predicted_class)
        worksheet.write(row, 2, score)

    workbook.close()




def load_model_from_pb_file():
    """" Load previously trained model in .pb format  Directory containing assets, variables, saved_model.pb"""
    global model
    # pojedynczyfolder
    filename = filedialog.askdirectory(mustexist=True)
    if not filename:
        return


    label1.configure(text=f"{filename} loading...")
    model = load_from_pb(filename)  # long process
    print(model.summary())
    label1.configure(text=f"{filename} loaded succesfully!")


def trening(activation, vocab_size, neurons, dropout):
    #print the paramteres
    print("activation:", activation, "dense recurrent neurons:", int(neurons), float(dropout))

    global sciezka_plikow_train
    global sciezka_plik_test
    global class_names
    global model

    if sciezka_plikow_train is None or sciezka_plik_test is None:
        label1.configure(text=f"Nie zaladowano danych.")
        return None

    #Loading the data
    train_source, test_source = str(sciezka_plikow_train), str(sciezka_plik_test)
    raw_train_ds, raw_val_ds, raw_test_ds, raw_test_ds_one_hot, class_names = load_data(train_source, test_source )

    #Creating the model
    model = build_classifier_model(dropout=float(dropout), num_classes=len(class_names), train_dataset=raw_train_ds,
                                   recurrent_neurons=int(neurons),
                                   activation=activation, vocab_size=int(vocab_size))
    #Training
    train_classifier(raw_train_ds, raw_val_ds, raw_test_ds, raw_test_ds_one_hot, model)


if __name__ == '__main__':
    window = tk.Tk()

    #Dialog
    label1 = tk.Label(window,
                      text="Załaduj pliki do treningu sieci lub wczytaj gotowy model.",
                      width=100, height=4,
                      fg="blue")
    label1.pack()

    #Data loading
    button_explore = tk.Button(window,
                               text="Dane treningowe",
                               command=browse_train_files)
    button_explore.pack()

    button_explore2 = tk.Button(window,
                                text="Dane testowe",
                                command=browse_test_files)
    button_explore2.pack()


    #Setting the parameters for training
    #Activation function
    label3 = tk.Label(text="Funkcja aktywacji")
    entry3 = tk.StringVar()
    combobox = ttk.Combobox(window, textvariable=entry3)

    #RELU Rectified linear unit
    combobox['values'] = ('relu', 'tanh', 'sigmoid')
    combobox.current(0)

    label3.pack()
    combobox.pack()

    #Size of the Vocabulary
    label4 = tk.Label(text="Vocab size")
    entry4 = tk.Entry()
    entry4.insert(tk.END, '1000')
    label4.pack()
    entry4.pack()

    #Reccurent neurons in two layers(Actually the number will be doubled for the first recurrent layer)
    label5 = tk.Label(text="Ilosc neuronow w warstwie glebokiej")
    entry5 = tk.Entry()
    entry5.insert(tk.END, '32')
    label5.pack()
    entry5.pack()

    #Droput rate
    label6 = tk.Label(text="Dropout")
    entry6 = tk.Entry()
    entry6.insert(tk.END, '0.2')
    label6.pack()
    entry6.pack()

    #Padding
    label7 = tk.Label(text=" ")
    label7.pack()

    button = tk.Button(
        command=lambda: trening(activation=entry3.get(), vocab_size=entry4.get(), neurons=entry5.get(),
                                dropout=entry6.get()),
        text="Trening",
        width=8,
        height=1,
        bg="blue",
        fg="yellow",
    )

    #Padding
    button.pack()
    label8 = tk.Label(text=" ")
    label8.pack()

    #Do some predictions. Upload the files
    label9 = tk.Label(window,
                      text="Predykcja dla pliku",
                      width=100, height=4,
                      )
    button9 = tk.Button(window,
                        text="Załaduj pliki do klasyfikacji",
                        command=make_prediction_for_given_files,
                        bg="green",
                        fg="yellow")

    #Padding
    label9.pack()
    button9.pack()

    #Use previously trained model
    label10 = tk.Label(window,
                       text="Wczytanie gotowego modelu",
                       width=100, height=4,
                       )
    button10 = tk.Button(window,
                         text="Load model",
                         command=load_model_from_pb_file,
                         )
    label10.pack()
    button10.pack()

    # Padding
    button.pack()
    label11 = tk.Label(text=" ")
    label11.pack()

    print(type(tk.Tk.mainloop))
    window.mainloop()

