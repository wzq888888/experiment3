import copy
import os
import numpy as np
import pickle
from scipy.misc import imread, imresize
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
from feature import NPDFeature
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

def get_pic(num_train=500):
    data_ = './datasets/original/'
    if os.path.exists(os.path.join(data_, 'dataset.pickle')):
        dataset = load(data_ + 'dataset.pickle')
    else:
        dataset = {}
        faceset = load_data(os.path.join(data_, 'face'))
        nonfaceset = load_data(os.path.join(data_, 'nonface'))
        dataset['face'] = preprocess_data(faceset)
        dataset['nonface'] = preprocess_data(nonfaceset)
        # pickle dataset
        save(dataset, data_ + 'dataset.pickle')

    num_feature = dataset['face'].shape[1]
    num_val = dataset['face'].shape[0] + dataset['nonface'].shape[0] - num_train
    num_train_face = num_train // 2
    num_train_nonface = num_train - num_train_face
    num_val_face = dataset['face'].shape[0] - num_train_face
    num_val_nonface = num_val - num_train_face

    X_train = np.ndarray(shape=(num_train, num_feature), dtype=np.float32)
    y_train = np.ones((num_train,))
    X_test = np.ndarray(shape=(num_val, num_feature), dtype=np.float32)
    y_test = np.ones((num_val,))

    X_train[:num_train_face] = dataset['face'][:num_train_face]
    X_train[num_train_face:] = dataset['nonface'][:num_train_nonface]
    y_train[num_train_face:] = -1
    X_test[:num_val_face] = dataset['face'][num_train_face:]
    X_test[num_val_face:] = dataset['nonface'][num_train_nonface:]
    y_test[num_val_face:] = -1

    return {
      'X_train': X_train, 'y_train': y_train,
      'X_test': X_test, 'y_test': y_test
    }

def load_data(data_):
    #Load image data
    image_files = os.listdir(data_)
    imageset = None
    image_index = 0
    for image in image_files:
        image_file = os.path.join(data_, image)
        try:
            img = imread(image_file).astype(float)
            '''
            if img.shape != (image_size, image_size, 3):
                raise Exception('Unexpected image shape: %s' % str(img.shape))
            '''
            if imageset is None:
                image_size = img.shape[0]
                imageset = np.ndarray(shape=(len(image_files), image_size,image_size, 3), 
                                    dtype=np.float32)
            imageset[image_index] = img
            image_index += 1
        except IOError as e:
            print('Could not read:', image_file, ':', e)
    num_image = image_index
    imageset = imageset[:num_image, :, :, :]

    return imageset

def preprocess_data(dataset):
    '''convert image to greyscale,resize to (24,24) and extract NPD feature'''
    # feature size: 165600
    num_feature = 165600
    num_sample = dataset.shape[0]
    dataset_processed = np.ndarray(shape=(num_sample, num_feature), 
                                    dtype=np.float32)
    for i in range(num_sample):
        img_grey = convert_to_grey(dataset[i])
        img_grey_resize = imresize(img_grey, (24,24))
        npdfeature = NPDFeature(img_grey_resize).extract()
        dataset_processed[i] = npdfeature
    return dataset_processed

def convert_to_grey(img):
    return img[:,:,0] * 0.3 + img[:,:,1] * 0.59 + img[:,:,2] * 0.11


def save(file, filename):
    with open(filename, 'wb') as f:
        pickle.dump(file, f)

def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def compute_accuracy(y_pred, y_groundture):
    return np.mean(np.array(y_pred == y_groundture), dtype = np.float32)


class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        self.weak_classifiers = []
        self.weak_classifier_weights = np.zeros(self.n_weakers_limit, dtype=np.float32)
        self.weak_classifier_errors = np.ones(self.n_weakers_limit, dtype=np.float32)

    def __str__(self):
        _clf = str(type(self.weak_classifier))[8:-2]
        _n = self.n_weakers_limit
        _str = 'AdaBoostClassifier(weak_classifier=%s, n_weakers_limit=%d)' % (_clf, _n)
        return _str


    def fit(self, X, y, verbose=True):
        '''Build a boosted classifier from the training set (X, y).
        '''

        num_train = X.shape[0]

        # initialize sample weight to 1.0/num_sample
        sample_weight = np.ones((num_train,)) / num_train

        # Clear any previous fit results
        self.weak_classifiers = []
        self.weak_classifier_weights = np.zeros(self.n_weakers_limit, dtype=np.float32)
        self.weak_classifier_errors = np.ones(self.n_weakers_limit, dtype=np.float32)

        for iboost in range(self.n_weakers_limit):
            # boost step
            sample_weight, classifier_weight, classifier_error = \
                self._boost(iboost, X, y, sample_weight)

            # early terminite
            if sample_weight is None:
                if verbose:
                    print('the error of current classifier as bad as random guessing(or worse), stop boost')
                break

            self.weak_classifier_weights[iboost] = classifier_weight
            self.weak_classifier_errors[iboost] = classifier_error

            # stop is error is 0
            if classifier_weight == 0:
                if verbose:
                    print('the error of current classifier is 0, stop boost')
                break

            if verbose:
                print('boost step %d / %d: classifier weight: %.3f, classification error: %.3f'
                      % (iboost + 1, self.n_weakers_limit, classifier_weight, classifier_error))

    def _boost(self, iboost, X, y, sample_weight):
        # create and train a new weak classifier
        clf = self._create_classifier()
        clf.fit(X, y, sample_weight=sample_weight)

        y_pred = clf.predict(X)

        incorrect = (y_pred != y)

        classifier_error = np.average(incorrect, weights=sample_weight)

        # stop if classifier is perfect
        if classifier_error == 0:
            return sample_weight, 1.0, 0.0

        # stop if classifier perform badly
        if classifier_error >= 0.5:
            self.classifiers.pop(-1)
            if len(self.classifiers) == 0:
                raise ValueError('weak_classifier provided can not used to create an emsemble classifier')
            return

        classifier_weight = 0.5 * np.log((1.0 - classifier_error) / classifier_error)

        zm = np.sum(sample_weight * np.exp(-classifier_weight * y * y_pred))
        sample_weight *= np.exp(-classifier_weight * y * y_pred)
        sample_weight /= zm

        return sample_weight, classifier_weight, classifier_error

    def _create_classifier(self):
        '''create a copy of self.weak_classifier'''
        clf = copy.deepcopy(self.weak_classifier)
        self.weak_classifiers.append(clf)
        return clf

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.
        '''
        scores = sum((clf.predict(X) * w for clf, w in zip(self.weak_classifiers, self.weak_classifier_weights)))
        return scores

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples
        '''
        scores = sum((clf.predict(X) * w for clf, w in zip(self.weak_classifiers, self.weak_classifier_weights)))
        y_pred = np.ones((X.shape[0],))
        y_pred[scores < threshold] = -1
        return y_pred

    def staged_predict(self, X, threshold=0):
        '''
        Yeild the ensemble predicrtion after after each boost.
        '''
        scores = None
        for clf, w in zip(self.weak_classifiers, self.weak_classifier_weights):
            current_score = clf.predict(X) * w  # (N,)

            if scores is None:
                scores = current_score
            else:
                scores += current_score

            current_pred = np.ones((X.shape[0],))
            current_pred[scores < threshold] = -1
            yield current_pred

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    # load dataset. (total 1000 samples, use num_train samples to fit adaboost model)
    num_train = 100
    dataset = get_pic(num_train=100)
    X_train, y_train, X_test, y_test = dataset['X_train'], dataset['y_train'], dataset['X_test'], dataset['y_test']
    print('datasize: 1000, use %d samples to fit adaboost model.' % num_train)
    max_depth = 2
    min_samples_leaf = 1
    min_samples_split = None
    n_weakers_limit = 50

    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    ada_model = AdaBoostClassifier(weak_classifier=dt, n_weakers_limit=n_weakers_limit)
    ada_model.fit(X_train, y_train,verbose=True)

    y_pred_train = ada_model.predict(X_train)
    y_pred_test = ada_model.predict(X_test)
    train_accuracy = compute_accuracy(y_pred_train, y_train)
    val_accuracy = compute_accuracy(y_pred_test, y_test)

    print('\ntraining acc: %.3f.\nvaluation acc: %.3f.' % (train_accuracy, val_accuracy))
    
    # Classification Report
    target_names = ['nonface', 'face']
    # wirte report into report.txt
    filename = 'report.txt'
    with open(filename, 'w') as f:
        target_names = ['nonface', 'face']
        f.write('training data:\n\n- Classifier: %s\n\n- Report:\n%s'
                % (ada_model, classification_report(y_train, y_pred_train, target_names=target_names)))
        f.write('\n\ntesting data:\n\n- Classifier: %s\n\n- Report:\n%s'
                % (ada_model, classification_report(y_test, y_pred_test, target_names=target_names)))

    # fit a raw tree
    d = DecisionTreeClassifier(max_depth=max_depth)
    d.fit(X_train, y_train)
    dt_raw_accuracy = d.score(X_test, y_test)
    print('training acc: %.5f' % d.score(X_train, y_train))
    print('test acc: %.5f' % dt_raw_accuracy)

    # fit a decision tree
    depth = None
    d = DecisionTreeClassifier(max_depth=depth)
    d.fit(X_train, y_train)
    dt_accuracy = d.score(X_test, y_test)
    print('training accuracy: %.5f' % d.score(X_train, y_train))
    print('test accuracy: %.5f' % dt_accuracy)

    # visualize result.
    # 
    # plot the ensemble prediction accuracy after each boost
    #
    ada_accuracy_train = np.zeros((n_weakers_limit,))
    for i, y_pred in enumerate(ada_model.staged_predict(X_train)):
        ada_accuracy_train[i] = compute_accuracy(y_pred, y_train)

    ada_accuracy_test = np.zeros((n_weakers_limit,))
    for i, y_pred in enumerate(ada_model.staged_predict(X_test)):
        ada_accuracy_test[i] = compute_accuracy(y_pred, y_test)

    # change the power of weak classifer (change tree depth)
    ada_models_acc = [] # store different AdaBoost model behaviors
    n_weakers = 55
    for dt_depth in range(1, 3):
        print('Fits a AdaBoost model with tree depth %d' %dt_depth)
        dt = DecisionTreeClassifier(max_depth=dt_depth)
        ada = AdaBoostClassifier(weak_classifier=dt, n_weakers_limit=n_weakers)
        ada.fit(X_train, y_train,verbose=False)
        accuracys = np.zeros((n_weakers,))
        for i, y_pred in enumerate(ada.staged_predict(X_test)):
            accuracys[i] = compute_accuracy(y_pred, y_test)
        print('accuracy: %.3f' % accuracys[-1])
        ada_models_acc.append(accuracys)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('train acc and test acc')

    ax.plot(np.arange(n_weakers_limit) + 1, ada_accuracy_train,
            label='AdaBoost Train',
            color='black')

    ax.plot(np.arange(n_weakers_limit) + 1, ada_accuracy_test,
            label='AdaBoost Test',
            color='red')

    ax.plot(np.arange(n_weakers_limit) + 1, [dt_raw_accuracy] * n_weakers_limit, 'k-', 
            label='weak classifier (depth=%d)'% max_depth)

    ax.plot(np.arange(n_weakers_limit) + 1, [dt_accuracy] * n_weakers_limit, 'k--', 
            label='Decision Tree (depth=%s)'% str(depth))
    
    ax.set_xlabel('n_classifiers')
    ax.set_ylabel('acc')

    leg = ax.legend(loc='lower right', fancybox=True)
    leg.get_frame().set_alpha(0.8)

    
    fig_2 = plt.figure()
    ax = fig_2.add_subplot(111)
    ax.set_title('AdaBoost Test acc with different weak decision tree classifier')


    ax.plot(np.arange(n_weakers) + 1, ada_models_acc[0],
            label='tree depth=1',
            color='blue')
    ax.plot(np.arange(n_weakers) + 1, ada_models_acc[1],
            label='tree depth=2',
            color='red')
    ax.plot(np.arange(n_weakers) + 1, ada_models_acc[2],
            label='tree depth=3',
            color='green')
    ax.plot(np.arange(n_weakers) + 1, ada_models_acc[3],
            label='tree depth=4',
            color='grey')
    ax.plot(np.arange(n_weakers) + 1, ada_models_acc[4],
            label='tree depth=5',
            color='black')
    
    ax.set_xlabel('n_classifiers')
    ax.set_ylabel('acc')
    leg = ax.legend(loc='lower right', fancybox=True)
    leg.get_frame().set_alpha(0.8)

    plt.show()