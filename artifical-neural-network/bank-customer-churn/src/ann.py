# Classification template

# Importing the libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

import training_set_import

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # mac issue


# Get dataset
dataset = training_set_import.get_dataset()

# Split dataset into dependent (y / output) and independent (x / input) values
x, y = training_set_import.split_dataset(dataset)

# Encode labels in dataset to numerical values - avoiding TODO - dummy field problem???
x = training_set_import.encode_independent_vars(x)

# Split dataset and scale values to reduce computation complexity (performance)
x_train, x_test, y_train, y_test = training_set_import.split_training_set(x, y)

# Scale dataset (hang on to the scale set for later predictions (it was fitted)
x_train, x_test, sc = training_set_import.scale_features(x_train, x_test)

# Basic training and validation by comparing against test set
def create_classifier():
    classifier = Sequential()
    classifier.add(Dense(input_dim=11, units=22, kernel_initializer='uniform', activation='relu'))
    # classifier.add(Dropout(rate=0.1))  # Prevents over fitting by disabling 10% of neurons
    classifier.add(Dense(units=22, kernel_initializer='uniform', activation='relu'))
    # classifier.add(Dropout(rate=0.1))  # Prevents over fitting by disabling 10% of neurons
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


# self.model.fit(x_train, y_train, batch_size, epochs)
# y_pred = classifier.predict(x_test)
# y_pred = (y_pred > 0.5)
#
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)

# Training using Cross value scoring (capable of working out mean and variance (batches up train set train and test into seperate steps
classifier = KerasClassifier(build_fn=create_classifier, batch_size=25, epochs=500)
accuracies = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=10, n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()


# Training using GridSearch for hyper paramater tuning
# def create_classifier(optimizer):
#     classifier = Sequential()
#     classifier.add(Dense(input_dim=11, units=6, kernel_initializer='uniform', activation='relu'))
#     classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
#     classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
#     classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
#     return classifier
#
#
# classifier = KerasClassifier(build_fn=create_classifier)
# parameters = {'batch_size': [25, 32], 'epochs': [100, 500], 'optimizer': ['adam', 'rmsprop']}
#
# grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)
# grid_search = grid_search.fit(x_train, y_train)
#
# best_parameters = grid_search.best_params_
# best_accuracy = grid_search.best_score_




