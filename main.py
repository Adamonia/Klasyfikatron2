from statistics import geometric_mean
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import xlsxwriter
from datetime import datetime


# from official.nlp import optimization  # to create AdamW optmizer


def load_data(train_source: str, test_source: str, batch_size=128):
    """Ladowanie danych z podzialem na treningowe, walidacyjne (80/20 %) i testowe"""

    batch_size = batch_size
    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_source,
        batch_size=batch_size,
        validation_split=0.2,
        subset="training",
        label_mode='categorical',
        seed=1337,
    )
    raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        train_source,
        batch_size=batch_size,
        validation_split=0.2,
        subset="validation",
        label_mode='categorical',
        seed=1337,
    )

    # Dane treningowe ladowane dwukrotnie w roznym formacie. Kodowanie numerem klasy
    raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
        test_source, batch_size=batch_size,
        label_mode='int'
    )
    # Kodowanie OneHot Encoding
    raw_test_ds_one_hot = tf.keras.preprocessing.text_dataset_from_directory(
        test_source, batch_size=batch_size,
        label_mode='categorical'
    )

    print(
        "Number of batches in raw_train_ds: %d"
        % tf.data.experimental.cardinality(raw_train_ds)
    )
    print(
        "Number of batches in raw_val_ds: %d" % tf.data.experimental.cardinality(raw_val_ds)
    )
    print(
        "Number of batches in raw_test_ds: %d"
        % tf.data.experimental.cardinality(raw_test_ds)
    )

    return raw_train_ds, raw_val_ds, raw_test_ds, raw_test_ds_one_hot, raw_train_ds.class_names


tf.get_logger().setLevel('ERROR')


# AUTOTUNE = tf.data.experimental.AUTOTUNE

# BERT
# bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8'
# tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
# tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2'


def build_classifier_model(dropout: float, num_classes: int, train_dataset, recurrent_neurons=32,
                           activation='relu', vocab_size=1000):
    """"Definicja modelu"""

    # Pierwsza warstwa - wektoryzacja - caly preprocessing
    # source: https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/TextVectorization
    # standardize each sample  lowercasing + punctuation stripping
    # split each sample into  words
    # index tokens (associate a unique int value with each token)
    # transform each sample using this index, either into a vector of ints or a dense float vector
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=vocab_size)
    encoder.adapt(train_dataset.map(lambda text, label: text))

    model = tf.keras.Sequential([
        encoder,
        # projecting the embeddings into 64 dimensional space
        tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
        # Recurrent Neurons
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(2 * recurrent_neurons, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(recurrent_neurons)),
        # Classification
        # projection into 64D space
        tf.keras.layers.Dense(32, activation=activation),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(num_classes, activation='sigmoid', name='classifier')
    ])

    return model


# def build_classifier_model_bert(dropout: float, num_classes: int, encoder_trainable=False, dense_neurons=512,
#                            activation='relu', train_dataset=None):
#
#
#     text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
#     preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
#     encoder_inputs = preprocessing_layer(text_input)
#
#     encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=encoder_trainable, name='BERT_encoder')
#     outputs = encoder(encoder_inputs)
#     net = outputs['pooled_output']
#
#     net = tf.keras.layers.Dense(dense_neurons, activation=activation, name='dense')(net)
#     net = tf.keras.layers.Dropout(dropout)(net)
#     net = tf.keras.layers.Dense(num_classes, activation='sigmoid', name='classifier')(net)
#     return tf.keras.Model(text_input, net)


def train_classifier(raw_train_ds, raw_val_ds, raw_test_ds, raw_test_ds_one_hot, model):
    # callback stoppping the training when there is no improvement
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    # Set the loss function and the optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    # train
    history = model.fit(x=raw_train_ds,
                        validation_data=raw_val_ds,
                        epochs=30, callbacks=[callback])

    # Save the model
    saved_model_path = './lstm_model'
    model.save(saved_model_path, include_optimizer=False)

    # testing
    test_model(model, raw_test_ds, raw_test_ds_one_hot)

    return model, history


def test_model(classifier_model, raw_test_ds, raw_test_ds_one_hot):
    # making predictions and calculating the metrics
    loss, accuracy, recall, precision = classifier_model.evaluate(raw_test_ds_one_hot)

    # make predictions - all probabilities
    predictions = classifier_model.predict(raw_test_ds)
    # final predictions
    predicted_class = np.argmax(predictions, axis=1)

    # y - correct labels
    y = []
    for utterance in raw_test_ds.unbatch():
        y.append(utterance[1])

    confusion_matrix = sklearn.metrics.confusion_matrix(y, predicted_class)

    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')
    print(f'Recall: {recall}')
    print(f'Precision: {precision}')

    f1 = geometric_mean([precision, recall])
    print(f'F1score: {f1}')

    save_the_test_report(predictions, accuracy, recall, precision, f1)


def save_the_test_report(predictions, accuracy, recall, precision, f1):
    # Saving the report
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")

    workbook = xlsxwriter.Workbook('report_{}.xlsx'.format(dt_string))
    worksheet = workbook.add_worksheet()

    # conf_converted = np.array2string(confusion_matrix, precision=1, separator=',',
    #                   suppress_small=True)

    conf_converted = np.array(confusion_matrix).__str__()

    worksheet.set_column('E:E', 7 * len(predictions[0]))

    cell_format = workbook.add_format({'bold': True})
    cell_format.set_bg_color('#778899')
    cell_format.set_font_size(12)
    cell_format.set_align('center')
    cell_format.set_align('vcenter')

    cell_format2 = workbook.add_format({'bold': False})
    cell_format2.set_bg_color('#DCDCDC')
    cell_format2.set_text_wrap()
    cell_format2.set_align('center')
    cell_format2.set_align('vcenter')

    cell_format3 = workbook.add_format({'bold': False})
    cell_format3.set_bg_color('#C0C0C0')
    cell_format3.set_align('center')
    cell_format3.set_align('vcenter')

    worksheet.write('A1', 'Accuracy', cell_format)
    worksheet.write('B1', 'Precision', cell_format)
    worksheet.write('C1', 'Recall', cell_format)
    worksheet.write('D1', 'F1score', cell_format)
    worksheet.write('E1', 'Confusion', cell_format)

    worksheet.write('A2', accuracy, cell_format2)
    worksheet.write('B2', precision, cell_format3)
    worksheet.write('C2', recall, cell_format2)
    worksheet.write('D2', f1, cell_format3)
    worksheet.write('E2', conf_converted, cell_format2)

    workbook.close()


def make_predictions(model, text, class_names=None):
    #use the model (convert strings to Tensors)
    predictions = model.predict(tf.constant(text))
    #logits to class
    predicted_class = np.argmax(predictions, axis=1)
    #getting the highest score
    scores = np.max(predictions, axis=1)

    if class_names:
        #if class_names were provided
        return predictions, list(map(predicted_class, lambda prediction: class_names[prediction])), scores
    return predictions, predicted_class, scores


def load_from_pb(dir):
    """" Load a ready to go model for classification. Directory containing assets, variables, saved_model.pb"""
    return tf.keras.models.load_model(dir, options=tf.saved_model.LoadOptions(
        experimental_io_device='/job:localhost'
    ))

#
# if __name__ == "__main__":
#     train_source, test_source = "aclImdb/train", "aclImdb/test"
#     raw_train_ds, raw_val_ds, raw_test_ds, raw_test_ds_one_hot, class_names = load_data(train_source, test_source)
#     train_classifier(raw_train_ds, raw_val_ds, raw_test_ds, raw_test_ds_one_hot, class_names)
