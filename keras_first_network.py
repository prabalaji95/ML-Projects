# first neural network with keras tutorial
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

#split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]

#The model expects rows of data with 8 variables (the input_shape=(8,) argument).
#The first hidden layer has 12 nodes and uses the relu activation function.
#The second hidden layer has 8 nodes and uses the relu activation function.
#The output layer has one node and uses the sigmoid activation function.

#define the keras model
model = Sequential()
#The most confusing thing here is that the shape of the input to the model is defined as an argument
#on the first hidden layer. This means that the line of code that adds the first Dense layer is
#doing two things, defining the input or visible layer and the first hidden layer.
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#compile the keras model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])



#Training occurs over epochs, and each epoch is split into batches.
#poch: One pass through all of the rows in the training dataset
#Batch: One or more samples considered by the model within an epoch before weights are updated

#fit the keras model on the dataset
model.fit(X, y, epochs = 150, batch_size = 10)

#evaluate your keras model 
_, accuracy = model.evaluate(X, y)
print("Accuracy = %.2f" % (accuracy*100))


#Make probability predictions with the model 
predictions = model.predict(X)
#round predictions
rounded = [round(x[0]) for x in predictions]


# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)

# summarize the first 5 cases
for i in range(5):
 print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))