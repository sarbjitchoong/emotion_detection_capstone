# data visualization functions

#libraries imported
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os


# function to show sample photo
def sampleimage(basewidth, image1):
    basewidth = basewidth                                 # define the size or the height of the image
    img = Image.open(image1)                              # open the image
    wpercent = (basewidth/float(img.size[0]))             
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize))                  
    img.save('somepic.jpg')                               # saving the new image
    return Image.open("somepic.jpg")                      # opening the image



# function for the plot
# got this code from tensorflow image classification

def plotting(history, epochs):                            # define the history/ where the model was saved, number of epochs
    epochs=epochs
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(epochs)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    return plt.show()







# function to plot the predictions
# got this code from tensorflow image classification

def prediction_visuals(image, model, class_names):             # define the image variable, model used, class_names variable  
    img = tf.keras.utils.load_img(
        image, target_size=(48, 48))                           # loading the image and resizing to 48 x 48 pixels
    img_array = tf.keras.utils.img_to_array(img)               # converting to array
    img_array = tf.expand_dims(img_array, 0)                   # creating a batch
    
    predictions = model.predict(img_array)                     # prediction of the image
    score = tf.nn.softmax(predictions[0])
    return print(
    '\033[1m' +  "This image most likely belongs to {} ."      # printing final result
    .format(class_names[np.argmax(score)]))






# function to plot sample images from the dataset
# got this code from tensorflow image classification

def plotting_visualization(data,class_names):                  # define the image dataset, and the class name variable
    plt.figure(figsize=(15, 15))   
    for images, labels in data.take(1):
        for i in range(15):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    return plotting_visualization






# function to plot the newly augmented photos
# got this code from tensorflow image classification

def augmented_data(data, data_augmentation):                    # define the image dataset, and the data augmentation variable
    tf.get_logger().setLevel('ERROR')                           # added an error remover
    for image, _ in data.take(1):
        plt.figure(figsize=(8, 8))
        first_image = image[0]
    for i in range(6):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')
    return augmented_data
        
        
        
        
        
        
        
        
        
