import cv2
import numpy as np
import glob

# load training data
image_array = np.zeros((1, 50400))
label_array = np.zeros((1, 3), 'float')
training_data = glob.glob('training_data3/*.npz')

# collecting all training data and labels
for single_training in training_data:
    with np.load(single_training) as data:
        train_temp = data['train']
        train_labels_temp = data['train_labels']

        # retrieve the single training data and label
        print train_temp.shape
        print train_labels_temp.shape
    image_array = np.vstack((image_array, train_temp))
    label_array = np.vstack((label_array, train_labels_temp))

# get rid of first row of zeroes
train = image_array[1:, :]
train_labels = label_array[1:, :]
print train.shape
print train_labels.shape

e1 = cv2.getTickCount()

# create ANN
layer_sizes = np.int32([50400, 32, 3])
model = cv2.ANN_MLP()
model.create(layer_sizes)
criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 0.0001)
criteria2 = (cv2.TERM_CRITERIA_COUNT, 100, 0.001)
params = dict(term_crit = criteria,
              train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,
              bp_dw_scale = 0.001,
              bp_moment_scale = 0.0)

print 'Training Neural Network.....'
num_iter = model.train(train, train_labels, None, params = params)

e2 = cv2.getTickCount()
time_duration = (e2 - e1)/cv2.getTickFrequency()
print 'Training Finish! Duration: ', time_duration
print 'Ran for %d iterations' % num_iter

# save params
model.save('ann_param/ann.xml')

# test accuracy on our training data/ later test on test data (data we did not use for training)
ret, resp = model.predict(train)
prediction = resp.argmax(-1)
print 'Prediction: ', prediction

correct_labels = train_labels.argmax(-1)
print 'Correct Labels: ', correct_labels

print 'Test Accuracy on Training data'
train_rate = np.mean(prediction == correct_labels)
print 'Train accuracy: %f: ' %(train_rate*100)

