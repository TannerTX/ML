import os
from reformat import Formatter
from PIL import Image
import random as rd
import shutil as sh
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

class Model_Producer:
    
    def __init__(self, IMG_DIMS, IMG_FORMAT, IMG_DIRS, POSTPROCESSED_DIR, OUTPUT_NM, EPOCH):
        self.IMG_DIMS = IMG_DIMS
        self.IMG_FORMAT = IMG_FORMAT
        self.SRC_IMG_DIRS = IMG_DIRS # ** List **
        self.POSTPROCESSED_DIR = POSTPROCESSED_DIR
        self.OUTPUT_NM = OUTPUT_NM
        self.EPOCHS = EPOCH
    
    def clean_slate(self):
        print('*****[Cleaning Up]*****')
        if os.path.exists('train'):
            sh.rmtree('train')
            print("Removed Training Data Directory")

        if os.path.exists('test'):
            sh.rmtree('test')
            print("Removed Test Data Directory")

        for DIR in self.SRC_IMG_DIRS:

            contents = os.listdir(DIR)

            if contents:
                for file in contents:
                    if os.path.isfile(os.path.join(DIR, file)):
                        os.remove(os.path.join(DIR, file))
                    elif os.path.isdir(os.path.join(DIR, file)):
                        sh.rmtree(file)
                print(f"Cleaned Image Source")


        contents = os.listdir(self.POSTPROCESSED_DIR)
        
        if contents:
            for file in contents:
                if os.path.isfile(os.path.join(self.POSTPROCESSED_DIR, file)):
                    os.remove(os.path.join(self.POSTPROCESSED_DIR, file))
                elif os.path.isdir(os.path.join(self.POSTPROCESSED_DIR, file)):
                    sh.rmtree(file)
            print("Removed Postprocessed Data")


    def remove_corrupted_imgs(self, primary):

        if primary == True:
            DIRS = [self.SRC_IMG_DIRS[0], self.SRC_IMG_DIRS[1]]
        else:
            DIRS = ['./train/Dog', './train/Cat', './test/Dog', './test/Cat']


        for DIR in DIRS:
            for filename in os.listdir(DIR):
                try:
                    im = Image.open(DIR + "/" + filename)
                    im.verify()
                except:
                    print(f"{filename} IS CORRUPTED")
                    os.remove(DIR + '/' + filename)
    

    def gen_datasets(self):
        TRAIN_PERCENT = 0.8
        filenames = []
    
        for file in os.listdir(self.POSTPROCESSED_DIR):
            if file.endswith('.jpg'):
                filenames.append(file)
    
        num_train = int(len(filenames) * TRAIN_PERCENT)
    
        rd.shuffle(filenames)
    
        train_filenames = filenames[:num_train]
        test_filenames = filenames[num_train:]
    
        if not os.path.exists('train'):
            os.makedirs('train')
            os.makedirs(os.path.join('train', 'Cat'))
            os.makedirs(os.path.join('train', 'Dog'))


        if not os.path.exists('test'):
            os.makedirs('test')
            os.makedirs(os.path.join('test', 'Cat'))
            os.makedirs(os.path.join('test', 'Dog'))
    
        for file in train_filenames:
            if file.startswith('cat'):
                os.rename(os.path.join(self.POSTPROCESSED_DIR, file), os.path.join('train/Cat', file))
            else:
                os.rename(os.path.join(self.POSTPROCESSED_DIR, file), os.path.join('train/Dog', file))
            
    
        for file in test_filenames:
            if file.startswith('cat'):
                os.rename(os.path.join(self.POSTPROCESSED_DIR, file), os.path.join('test/Cat', file))
            else:
                os.rename(os.path.join(self.POSTPROCESSED_DIR, file), os.path.join('test/Dog', file))
            

    def load_images(self):
        OG_SRC_CAT = './PetImages/Cat'
        OG_SRC_DOG = './PetImages/Dog'

        for file in os.listdir(OG_SRC_CAT):
            sh.copy(os.path.join(OG_SRC_CAT, file), self.SRC_IMG_DIRS[0])

        for file in os.listdir(OG_SRC_DOG):
            sh.copy(os.path.join(OG_SRC_DOG, file), self.SRC_IMG_DIRS[1])


    def process_images(self):

        CAT_DIR = self.SRC_IMG_DIRS[0]
        DOG_DIR = self.SRC_IMG_DIRS[1]
        
        # Handle Cat Imgs
        formatter = Formatter(self.IMG_DIMS, 'jpg', CAT_DIR)
        formatter.batch_rename('cat')
        formatter.resize(CAT_DIR, self.POSTPROCESSED_DIR)
        # Handle Dog Imgs
        formatter = Formatter(self.IMG_DIMS, 'jpg', DOG_DIR)
        formatter.batch_rename('dog')
        formatter.resize(DOG_DIR, self.POSTPROCESSED_DIR)

    def train_model(self):
        TRAINING_DIR = './train'
        TESTING_DIR = './test'

        train_datagen = ImageDataGenerator(rescale = 1/255)
        test_datagen = ImageDataGenerator(rescale = 1/255)

        training_imgs = train_datagen.flow_from_directory(TRAINING_DIR, target_size=self.IMG_DIMS, batch_size=32, class_mode='binary')
        testing_imgs = train_datagen.flow_from_directory(TESTING_DIR, target_size=self.IMG_DIMS, batch_size=32, class_mode='binary')


        model = models.Sequential()
        model.add( layers.Conv2D(16, (3, 3), activation='relu', input_shape=(self.IMG_DIMS[0], self.IMG_DIMS[1], 3)) )
        model.add( layers.MaxPooling2D((2, 2)) )
        model.add( layers.Conv2D(32, (3, 3), activation='relu') )
        model.add( layers.MaxPooling2D((2, 2)) )
        model.add( layers.Conv2D(32, (3, 3), activation='relu') )
        model.add( layers.MaxPooling2D((2, 2)) )
        model.add( layers.Conv2D(32, (3, 3), activation='relu') )
        model.add( layers.MaxPooling2D((2, 2)) )
        model.add( layers.Conv2D(64, (3, 3), activation='relu') )
        model.add( layers.MaxPooling2D((2, 2)) )
        model.add( layers.Conv2D(64, (3, 3), activation='relu') )

        model.add( layers.Flatten() )
        model.add( layers.Dense(512, activation='relu') )
        model.add( layers.Dense(1, activation='sigmoid') )

        model.compile( optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )
        model.fit( training_imgs, epochs=self.EPOCHS, validation_data=testing_imgs )

        loss, accuracy= model.evaluate(testing_imgs)
        print(f"Loss: {loss} | Accuracy: {accuracy}")

        model.save(self.OUTPUT_NM)

    def predict(self, model_NM):
        model = models.load_model(model_NM)
        num_pics = len(os.listdir('./predict_images'))
        f, axarr = plt.subplots(num_pics, 1, figsize=(15,15)) 
        f.tight_layout(pad=2.0)

        for idx, file in enumerate(os.listdir('./predict_images')):
            img = image.load_img(f'./predict_images/{file}')
            X = image.img_to_array(img)
            X = np.expand_dims(X, axis=0)
            images = np.vstack([X])
            res = model.predict(images)

            im = cv.imread(f'./predict_images/{file}')
            im_resized = cv.resize(im, (250, 250), interpolation=cv.INTER_LINEAR)

            axarr[idx].imshow(cv.cvtColor(im_resized, cv.COLOR_BGR2RGB))
            print(f"IMAGE {idx}: {res}")

            if res == 0:
                axarr[idx].set_title(f'{idx}: CAT', )
                # print(f"IMAGE #{idx}: CAT")
            elif res == 1:
                axarr[idx].set_title(f'{idx}: DOG')
                # print(f"IMAGE #{idx}: DOG")
            else:
                axarr[idx].set_title(f'{idx}: ERR')

        plt.show()

    def run(self):
        print('*****[Cleaning Up]*****')
        self.clean_slate()

        print('*****[Piping Images into SRC Directories]*****')
        self.load_images()
        
        print("*****[Checking For Corrupted Images]*****")
        self.remove_corrupted_imgs(primary=True)

        print("*****[Processing Images]*****")
        self.process_images()

        print("*****[Generating Datasets]*****")
        self.gen_datasets()

        print("*****[Checking Datasets for Corruption]*****")
        self.remove_corrupted_imgs(primary=False)

        print("*****[TRAINING MODEL]*****")
        self.train_model()
        
        print("Finished!")
