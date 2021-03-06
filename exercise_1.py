# Exercise 1

#make sure our experiment in keras is reproducible
import numpy
numpy.random.seed(42)

import argparse
import pandas
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from tensorflow import set_random_seed
set_random_seed(42)

#now, import keras 
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras import optimizers, regularizers


def read_args():
    parser = argparse.ArgumentParser(description='Exercise 1')
    # Here you have some examples of classifier parameters. You can add
    # more arguments or change these if you need to.
    parser.add_argument('--h_units', nargs='+', default=[1], type=int,
                        help='Number of hidden units of each hidden layer.')
    parser.add_argument('--dropout', nargs='+', default=[0], type=float,
                        help='Dropout ratio for every layer.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of instances in each batch.')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment, used in the filename'
                             'where the results are stored.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--reg_l2', type=float, default=0,
                        help='L2 regularization parameter')

    args = parser.parse_args()

    return args


def load_dataset():
    dataset = load_files('dataset/txt_sentoken', shuffle=False)

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=42)

    print('Training samples {}, test_samples {}'.format(
        len(X_train), len(X_test)))

    # TODO 1: Apply the Tfidf vectorizer to create input matrix
    vectorizer = TfidfVectorizer(
        binary=True,
        min_df=9,
        max_df=0.5,
        ngram_range=(1, 2),
        use_idf=True
    )

    
    vectorizer = vectorizer.fit(X_train)
    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)
    
    print("X_train shape = " + str(X_train.shape))

    return X_train, X_test, y_train, y_test


def main():
    args = read_args()
    X_train, X_test, y_train, y_test_orginal = load_dataset()

    # TODO 2: Convert the labels to categorical
    y_test_orginal_categ = keras.utils.to_categorical(y_test_orginal, 2)
    y_train_categ = keras.utils.to_categorical(y_train, 2)

    print()
    # TODO 3: Build the Keras model
    model = Sequential()
    # Add all the layers
    
    units_output_layer = y_test_orginal_categ.shape[1] #one per class

    if len(args.h_units) > 0:
        # Input to hidden layer 
        model.add(Dense(args.h_units[0], input_dim=X_train.shape[1], kernel_regularizer=regularizers.l2(args.reg_l2)))
        model.add(Dropout(args.dropout[0]))
        model.add(Activation('relu'))

        for i in range(1, len(args.h_units)):
            model.add(Dense(args.h_units[i],  input_dim=X_train.shape[1], kernel_regularizer=regularizers.l2(args.reg_l2)))
            model.add(Dropout(args.dropout[i]))
            model.add(Activation('relu'))
        
        # Hidden to output layer
        model.add(Dense(units_output_layer))
        model.add(Activation('softmax'))
    else:
        dropout = 0
        if len(args.dropout) > 0:
            dropout = args.dropout[0]
        model = Sequential([(Dense(units_output_layer, input_shape=(X_train.shape[1], ))),
                            Activation('softmax')   
                          ])

    print(model.summary())

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adagrad(), 
                  metrics=['accuracy']) 
    
    
    # TODO 4: Fit the model
    history = model.fit(X_train, y_train_categ, batch_size=args.batch_size, epochs=args.epochs)

    # TODO 5: Evaluate the model, calculating the metrics.
    # Option 1: Use the model.evaluate() method. For this, the model must be
    # already compiled with the metrics.
    #performance = model.evaluate(X_test)

    # Option 2: Use the model.predict() method and calculate the metrics using
    # sklearn. We recommend this, because you can store the predictions if
    # you need more analysis later. Also, if you calculate the metrics on a
    # notebook, then you can compare multiple classifiers.
    predictions = model.predict(X_test)
    
    for sample in predictions:
        if sample[0] < 0.5:
            sample[0] = 0
        else:
            sample[0] = 1

        if sample[1] < 0.5:
            sample[1] = 0
        else:
            sample[1] = 1
    
    acc = accuracy_score(predictions, y_test_orginal_categ)
    
    print('accuracy: ' + str(acc))
    predictions = numpy.argmax(predictions, axis=None, out=None)
    # TODO 6: Save the results.
    results = pandas.DataFrame(y_test_orginal, columns=['y_test_orginal'])
    results.loc[:, 'predicted'] = predictions
    if args.experiment_name is None or args.experiment_name == "":
        args.experiment_name = str(args.batch_size) + "-" + str(args.epochs) + "-" + str(args.h_units) + "-" + str(args.reg_l2) + "-" + str(args.dropout)
    results.to_csv('predictions_{}.csv'.format(args.experiment_name),
    index=False)

    score_file  = open('score_{}.txt'.format(args.experiment_name), 'w')
    score_file.write(str(acc) + '\n')
    score_file.close()


    

if __name__ == '__main__':
    main()