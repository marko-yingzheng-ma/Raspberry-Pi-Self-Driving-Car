import cv2
import numpy as np
import glob

# load testing data
image_array = np.zeros((1, 38400))
label_array = np.zeros((1, 3), 'float')
testing_data = glob.glob('testing_data/*.npz')

for single in testing_data:
    with np.load(single) as data:
        test_temp = data['train']
        test_labels_temp = data['train_labels']
    image_array = np.vstack((image_array, test_temp))
    label_array = np.vstack((label_array, test_labels_temp))

test = image_array[1:, :]
test_labels = label_array[1:, :]

# create ANN
layer_sizes = np.int32([50400, 32, 3])
model = cv2.ANN_MLP()
model.create(layer_sizes)
model.load('ann_param/ann.xml')

# generate prediction
ret, resp = model.predict(test)
print resp
prediction = resp.argmax(-1)
print prediction
true_labels = test_labels.argmax(-1)

test_rate = np.mean(prediction == true_labels)
print 'Test accuracy: %f' %(test_rate*100)