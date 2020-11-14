import tensorflow as tf
import numpy as np

# create model
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1)])

# define optimization algorithm and loss function
model.compile(optimizer='sgd',loss='mean_squared_error')

# define data
x = np.array([-1,0,1,2,3,4],dtype=int)
y = np.array([3,4,5,6,7,8],dtype=int)

# train model
model.fit(x,y,epochs=500)

# predict value
print(model.predict([10]))