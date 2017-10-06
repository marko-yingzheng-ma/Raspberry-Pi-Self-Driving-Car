# Raspberry-Pi-Self-Driving-Car

Car:
-	Stream_client.py used to setup client socket to stream video from PiCamera to the pipeline so that computer can receive
-	Run Python Flask controller.py web service for listening to any http request and steering the car accordingly 
o	Ex. http://192.168.43.141:5000/right. Make http request to this url will make the car turn right

Computer:
-	Collect_training.py used to setup server socket to receive upcoming video stream data from the car
-	Process the stream data in real time and used PyGame to detect keypress so that human driver can drive the car. 
-	When human drive press a key to drive, we save the key and the corresponding frame into nump array for later neural network use
-	After getting all the training data and storing them in training_data3 as a bunch of .npz file, in ann_training.py, we stich those separated files into a training_data_size * 50400 (120 X 420 pixels) numpy matrix and a training_data_size * 3 (3 kinds of steering, left, right ,forward. Ex. 1 0 0 means left is activated). 
-	Once we get our training data and labels, we created a cv2 neural net model as 50400 size for input layer, 32 nodes in hidden layer, and 3 nodes in output layer. 
-	We later specified which criteria (training params) we use for training.
-	After the training is done, we save the training parameters in ann.xml.
-	Once we have the ann.xml, we will run pi_driver.py on PC again to open socket server and receive video stream from Pi again. In each frame, we convert it to numpy array, feed it into our neural network with loaded params from ann.xml, and make prediction as either left, right, or forward. Then we send http request to corresponding url for the car to drive per prediction.
-	Training video is on Youtube: https://youtu.be/S7Y9lsmi2_o
