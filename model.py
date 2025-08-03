import tensorflow as tf
from tensorflow import keras
(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
print(x_train.shape)
print(x_test.shape)
x_train=x_train/255
x_test=x_test/255
x_train_flatten=x_train.reshape(60000,28*28)
x_test_flatten=x_test.reshape(10000,28*28)
model=keras.Sequential([keras.layers.Dense(100,input_shape=(784,),activation='relu'),keras.layers.Dense(10,activation='sigmoid')])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train_flatten,y_train,epochs=10)
model.evaluate(x_test_flatten,y_test)
y_pred=model.predict(x_test_flatten)
model.save('model.h5')