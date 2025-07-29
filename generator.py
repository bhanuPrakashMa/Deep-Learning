import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import random
import skimage.transform as st
import math

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.img_size = image_size
        self.rot = rotation
        self.mir = mirroring
        self.shuffle = shuffle
        
        self.current_index = 0
        self.num_batches = None
        self.curre_epoch = 0
        lbl = open(self.label_path)

        self.json_file_data = json.load(lbl)
        self.num_images = len(self.json_file_data)
        self.img_names = list(self.json_file_data.keys())



    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method



        if self.shuffle:
            random.shuffle(self.img_names)

        last_index = self.current_index + self.batch_size



        if last_index <= self.num_images:




            batch_img_names = self.img_names[self.current_index : last_index]
            self.current_index = last_index



        else:



            batch_img_names = self.img_names[self.current_index : self.num_images]
            batch_space_left = self.batch_size - (self.num_images - self.current_index)
            if self.shuffle:


                temp_list = self.img_names[0:self.current_index]
                random.shuffle(temp_list)
                self.img_names = temp_list + batch_img_names
            
            
            batch_img_names.extend(self.img_names[0:batch_space_left])
            self.current_index = batch_space_left
            self.curre_epoch += 1



        img_in_tuple = []
        label_in_tuple = []



        for i in batch_img_names:



            img = np.load(self.file_path + '\\' + i + '.npy')
            img = st.resize(img, self.img_size)
            img = self.augment(img)
            img_in_tuple.append(img)
            
            
            
            label_in_tuple.append(self.json_file_data[i])
        
        
        
        
        batch_tuple = (np.asarray(img_in_tuple), np.asarray(label_in_tuple))


        
        
        return batch_tuple
        #return images, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        
        
        #TODO: implement augmentation function
        if self.mir:


            if (random.randint(1,5)  > 0.75):
               img = np.flip(img)
        
        if self.rot:


            if (random.random() > 0.75):
                
                for i in range(random.randint(1,3)):
                    
                    img = np.rot90(img)

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.curre_epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict.get(x)
    
    
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        
        batch_images = self.next()
        images_in_batch = batch_images[0]
        labels_in_batch = batch_images[1]

        
        imgs_in_row = math.ceil(len(labels_in_batch) / 3) 
        imgs_in_col = 3

        figure = plt.figure()
        for i in range(len(labels_in_batch)):
        
            figure.add_subplot(imgs_in_row,imgs_in_col,i+1)
            plt.imshow(images_in_batch[i])
            plt.axis('off')
            plt.title(self.class_name(labels_in_batch[i]))
        
        
        plt.show()
                       
