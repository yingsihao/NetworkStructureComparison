import scipy.io as sio
import numpy as np
import pickle



def load_data(mat_file_path, width=28, height=28, max_=None, verbose=True):
    ''' Load data in from .mat file as specified by the paper.
        Arguments:
            mat_file_path: path to the .mat, should be in sample/
        Optional Arguments:
            width: specified width
            height: specified height
            max_: the max number of samples to load
            verbose: enable verbose printing
        Returns:
            A tuple of training and test data, and the mapping for class code to ascii value,
            in the following format:
                - ((training_images, training_labels), (testing_images, testing_labels), mapping)
    '''
    # Local functions
    def rotate(img):
        # Used to rotate images (for some reason they are transposed on read-in)
        flipped = np.fliplr(img)
        return np.rot90(flipped)

    def display(img, threshold=0.5):
        # Debugging only
        render = ''
        for row in img:
            for col in row:
                if col > threshold:
                    render += '@'
                else:
                    render += '.'
            render += '\n'
        return render

    # Load convoluted list structure form loadmat
    mat = sio.loadmat(mat_file_path)

    # Load char mapping
    mapping = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]}
    #pickle.dump(mapping, open('bin/mapping.p', 'wb' ))

    # Load training data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][0][0][0][0])
    training_images = mat['dataset'][0][0][0][0][0][0][:max_].reshape(max_, height, width)
    training_labels = mat['dataset'][0][0][0][0][0][1][:max_].reshape(max_)

    # Load testing data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][1][0][0][0])
    else:
        max_ = int(max_ / 6)
    testing_images = mat['dataset'][0][0][1][0][0][0][:max_].reshape(max_, height, width)
    testing_labels = mat['dataset'][0][0][1][0][0][1][:max_].reshape(max_)

    # Reshape training data to be valid
    if verbose == True: _len = len(training_images)
    for i in range(len(training_images)):
        #if verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r'))
        training_images[i] = rotate(training_images[i])
    if verbose == True: print('')

    # Reshape testing data to be valid
    if verbose == True: _len = len(testing_images)
    for i in range(len(testing_images)):
        #if verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r')
        testing_images[i] = rotate(testing_images[i])
    if verbose == True: print('')

    # Convert type to float32
    training_images = training_images.astype('float32')
    testing_images = testing_images.astype('float32')

    # Normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255

    nb_classes = len(mapping)

    #training_images = training_images.reshape(training_images.shape[0], height * width)
    #testing_images = testing_images.reshape(testing_images.shape[0], height * width)

    return ((training_images, training_labels), (testing_images, testing_labels), mapping, nb_classes)

train, test, x, y = load_data('emnist-letters.mat')
train_img, train_label = train

#print np.array(train_label).max()
#print x
#print y