import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import scipy

import warnings
warnings.filterwarnings('ignore')
import os
for dirname, _, _ in os.walk('D:\Projects\original dataset'):
    print(dirname)
train_path = "D:\Projects\original dataset\TRAIN"
test_path = "D:\Projects\original dataset\TEST"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import plot_model

from glob import glob

x_data = [] 
y_data = [] 

for category in glob(train_path+'/*'):
    for file in tqdm(glob(category+'/*')):
        img_array=cv2.imread(file)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        x_data.append(img_array) 
        y_data.append(category.split("/")[-1])
        
data=pd.DataFrame({'image': x_data,'label': y_data})

data.shape


from collections import Counter
Counter(y_data)


# Sample data and labels
labels = ['Organic', 'Recyclable']
sizes = [30, 70]  # Replace with your actual data

colors = ['#a0d157', '#c48bb8']
explode = (0.05, 0.05)  # Explode slices

# Create the pie chart
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%0.2f%%', startangle=90)

# Equal aspect ratio ensures that pie is drawn as a circle.
plt.axis('equal')

# Show the pie chart
plt.show()

plt.figure(figsize=(10, 10))
num_samples = min(9, len(data))  # Limit to the number of available samples

for i in range(num_samples):
    index = np.random.choice(len(data))  # Randomly select an index
    plt.subplot(3, 3, i + 1)
    
    try:
        plt.title('This image is of {0}'.format(data.label.iloc[index]), fontdict={'size':    5, 'weight': 'bold'})
        plt.imshow(data.image.iloc[index])
        plt.tight_layout()
    except Exception as e:
        print(f"Error processing index {index}: {str(e)}")

plt.show()


className = glob(train_path + '/*' )
numberOfClass = len(className)
print("Number Of Class: ",numberOfClass)



import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout

# Define input shape
input_shape = (224, 224, 3)

# Input layer
input_layer = Input(shape=input_shape)

# Convolutional layers
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D()(x)

x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D()(x)

x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D()(x)

# Flatten and fully connected layers
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)

# Output layer
output_layer = Dense(numberOfClass, activation='sigmoid')(x)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print model summary
model.summary()

batch_size = 256


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a simple model
model = Sequential()
model.add(Dense(64, input_shape=(10,)))
model.add(Dense(32))

# Create a text description of the model
model_description = "Simple Sequential Model"
model.summary(print_fn=lambda x: model_description)

# You can display the text description if needed
print(model_description)

# Save the model plot as an image
# You can customize the appearance using Matplotlib functions
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.text(0.5, 0.5, model_description, horizontalalignment='center', verticalalignment='center', fontsize=12)
plt.axis('off')  # Hide axes
plt.savefig('model_plot.png', bbox_inches='tight', pad_inches=0.1, format='png')
plt.show()



train_datagen = ImageDataGenerator(rescale= 1./255)
test_datagen = ImageDataGenerator(rescale= 1./255)


train_generator = train_datagen.flow_from_directory(
        train_path, 
        target_size= (224,224),
        batch_size = batch_size,
        color_mode= "rgb",
        class_mode= "categorical")

test_generator = test_datagen.flow_from_directory(
        test_path, 
        target_size= (224,224),
        batch_size = batch_size,
        color_mode= "rgb",
        class_mode= "categorical")

# Output layer with the correct number of units (numberOfClass)
output_layer = Dense(numberOfClass, activation='softmax')(x)  # Use 'softmax' for multi-class classification

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Now you can train the model
hist = model.fit_generator(
    generator=train_generator,
    epochs=10,
    validation_data=test_generator
)


plt.figure(figsize=[10,6])
plt.plot(hist.history["accuracy"], label = "Train acc")
plt.plot(hist.history["val_accuracy"], label = "Validation acc")
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(hist.history['loss'], label = "Train loss")
plt.plot(hist.history['val_loss'], label = "Validation loss")
plt.legend()
plt.show()

def predict_func(img): 
    plt.figure(figsize=(6,4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [-1, 224, 224,3])
    result = np.argmax(model.predict(img))
    if result == 0: print("\033[94m"+"This image -> Recyclable"+"\033[0m")
    elif result ==1: print("\033[94m"+"This image -> Organic"+"\033[0m")
    
import os

model_dir = "D:\WASTE CLASSIFICATION\dataset\DATASET\DATASET"

os.makedirs(model_dir, exist_ok=True)

model.save(os.path.join(model_dir, 'my_model.h5'))
print("Model saved as my_model.h5 in directory:", model_dir)