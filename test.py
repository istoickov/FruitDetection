from sklearn import svm
import numpy as np
import cv2
import os

def read_training_data(path_training):
    X_train_RGB, Y_train_RGB, X_train_HOG, Y_train_HOG = [], [], [], []

    print('Reading training data')
    print('Using RGB and HOG')

    for folder in os.listdir(path_training):
        print(folder + '/')

        for i, image in enumerate(os.listdir(os.path.join(path_training, folder))):
            path1 = os.path.join(path_training, folder)
            path1 = os.path.join(path1+'/', image)

            img = cv2.imread(path1)

            r = img.shape[0]
            c = img.shape[1]

            data = np.concatenate(img[0:r, 0:c, 2]).tolist() + np.concatenate(
                img[0:r, 0:c, 1]).tolist() + np.concatenate(img[0:r, 0:c, 0]).tolist()

            X_train_RGB.append(data)
            Y_train_RGB.append(folder)

            h = hog.compute(img)
            data = np.concatenate(h).tolist()
            print(data)
            X_train_HOG.append(data)
            Y_train_HOG.append(folder)

            if i == 30:
                break

    return X_train_RGB, Y_train_RGB, X_train_HOG, Y_train_HOG

def read_test_data(path_test):
    X_test_RGB, Y_test_RGB, X_test_HOG, Y_test_HOG = [], [], [], []

    print('Reading test data')
    print('Using RGB and HOG')

    for folder in os.listdir(path_test):
        for i, image in enumerate(os.listdir(os.path.join(path_test, folder))):
            path1 = os.path.join(path_test, folder)
            path1 = os.path.join(path1, image)

            img = cv2.imread(path1)

            data = []

            r = img.shape[0]
            c = img.shape[1]

            data = np.concatenate(img[0:r, 0:c, 2]).tolist() + np.concatenate(
                img[0:r, 0:c, 1]).tolist() + np.concatenate(img[0:r, 0:c, 0]).tolist()

            X_test_RGB.append(data)
            Y_test_RGB.append(folder)

            h = hog.compute(img)
            data = np.concatenate(h).tolist()
            X_test_HOG.append(data)
            Y_test_HOG.append(folder)

            if i == 10:
                break

    return X_test_RGB, Y_test_RGB, X_test_HOG, Y_test_HOG

def make_RGB_model(X_train_RGB, Y_train_RGB, X_test_RGB, Y_test_RGB):
    print('Making RGB SVM model')

    model_RGB = svm.SVC(decision_function_shape='ovo')
    model_RGB.fit(X_train_RGB, Y_train_RGB)

    print('Testing RGB SVM model')
    print('Accuracy:')
    print(model_RGB.score(X_test_RGB, Y_test_RGB))

def make_HOG_model(X_train_HOG, Y_train_HOG, X_test_HOG, Y_test_HOG):
    print('Training HOG SVM model')

    model_HOG = svm.SVC(decision_function_shape='ovo')
    model_HOG.fit(X_train_HOG, Y_train_HOG)

    print('Testing HOG SVM model')
    print('Accuracy:')
    print(model_HOG.score(X_test_HOG, Y_test_HOG))

if __name__ == '__main__':
    hog = cv2.HOGDescriptor()

    path_training = 'C:/Users/ivans/Desktop/MV project/Project/Dataset/Training/'
    path_test = 'C:/Users/ivans/Desktop/MV project/Project/Dataset/Test/'

    X_train_RGB, Y_train_RGB, X_train_HOG, Y_train_HOG = read_training_data(path_training)
    X_test_RGB, Y_test_RGB, X_test_HOG, Y_test_HOG = read_test_data(path_test)

    make_RGB_model(X_train_RGB, Y_train_RGB, X_test_RGB, Y_test_RGB)
    make_HOG_model(X_train_HOG, Y_train_HOG, X_test_HOG, Y_test_HOG)