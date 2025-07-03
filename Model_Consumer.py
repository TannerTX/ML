import os
import random as rd
import shutil as sh
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from reformat import Formatter
import functools
import time


class ModelProducer:
    def __init__(self, img_dims, img_format, img_dirs, postprocessed_dir, output_name, epochs):
        self.img_dims = img_dims
        self.img_format = img_format
        self.src_img_dirs = img_dirs
        self.postprocessed_dir = postprocessed_dir
        self.output_name = output_name
        self.epochs = epochs
        
        # Prioritize GPU
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU(s) detected: {[gpu.name for gpu in gpus]}")
            except RuntimeError as e:
                print(f"⚠️ GPU setup failed: {e}")
        else:
            raise RuntimeError("❌ No GPU found. TensorFlow will not run on GPU.")


    def log_step(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = func.__name__.replace('_', ' ').capitalize()
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"✅ Completed: {name} in {end - start:.2f}s")
            return result
        return wrapper

    def suppress_errors(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"❌ Error in {func.__name__}: {e}")
        return wrapper

    @log_step
    def clean_slate(self):
        
        for folder in ['train', 'test']:
            if os.path.exists(folder):
                sh.rmtree(folder)
                print(f"  - Removed '{folder}' directory")

        for dir_path in self.src_img_dirs:
            for item in os.listdir(dir_path):
                path = os.path.join(dir_path, item)
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    sh.rmtree(path)
            print(f"  - Cleaned image source: {dir_path}")

        for item in os.listdir(self.postprocessed_dir):
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                sh.rmtree(path)
        print("  - Cleared postprocessed images")

    @log_step
    def remove_corrupted_imgs(self, primary=True):

        dirs = (
            [self.src_img_dirs[0], self.src_img_dirs[1]]
            if primary else
            ['./train/Dog', './train/Cat', './test/Dog', './test/Cat']
        )

        for dir_path in dirs:
            for filename in os.listdir(dir_path):
                try:
                    with Image.open(os.path.join(dir_path, filename)) as im:
                        im.verify()
                except:
                    print(f"  - {filename} is corrupted and will be removed")
                    os.remove(os.path.join(dir_path, filename))
    
    @log_step
    def gen_datasets(self):

        train_percent = 0.8
        filenames = [f for f in os.listdir(self.postprocessed_dir) if f.endswith('.jpg')]
        rd.shuffle(filenames)

        num_train = int(len(filenames) * train_percent)
        train_filenames = filenames[:num_train]
        test_filenames = filenames[num_train:]

        for base in ['train', 'test']:
            for label in ['Cat', 'Dog']:
                os.makedirs(os.path.join(base, label), exist_ok=True)

        for file in train_filenames:
            dest = 'Cat' if file.startswith('cat') else 'Dog'
            sh.move(os.path.join(self.postprocessed_dir, file), os.path.join('train', dest, file))

        for file in test_filenames:
            dest = 'Cat' if file.startswith('cat') else 'Dog'
            sh.move(os.path.join(self.postprocessed_dir, file), os.path.join('test', dest, file))

        print(f"  - {len(train_filenames)} training images")
        print(f"  - {len(test_filenames)} testing images")

    @log_step
    def load_images(self):

        cat_src = './Dataset/Cat'
        dog_src = './Dataset/Dog'
        for file in os.listdir(cat_src):
            sh.copy(os.path.join(cat_src, file), self.src_img_dirs[0])
        for file in os.listdir(dog_src):
            sh.copy(os.path.join(dog_src, file), self.src_img_dirs[1])


    @log_step
    def process_images(self):

        for label, src_dir in zip(['cat', 'dog'], self.src_img_dirs):
            formatter = Formatter(self.img_dims, 'jpg', src_dir)
            formatter.batch_rename(label)
            formatter.resize(src_dir, self.postprocessed_dir)


    @log_step
    def train_model(self):

        train_dir = './train'
        test_dir = './test'

        train_gen = ImageDataGenerator(rescale=1 / 255)
        test_gen = ImageDataGenerator(rescale=1 / 255)

        train_imgs = train_gen.flow_from_directory(train_dir, target_size=self.img_dims, batch_size=32, class_mode='binary')
        test_imgs = test_gen.flow_from_directory(test_dir, target_size=self.img_dims, batch_size=32, class_mode='binary')

        model = models.Sequential([
            layers.Conv2D(16, (3, 3), activation='relu', input_shape=(*self.img_dims, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_imgs, epochs=self.epochs, validation_data=test_imgs)

        loss, acc = model.evaluate(test_imgs)
        model.save(self.output_name)
        print(f"  - Model saved as '{self.output_name}'")

    @log_step
    def predict(self, model_name):

        model = models.load_model(model_name)
        pred_dir = './predict_images'
        image_files = os.listdir(pred_dir)

        fig, axes = plt.subplots(len(image_files), 1, figsize=(15, 15))
        fig.tight_layout(pad=2.0)

        for idx, file in enumerate(image_files):
            img_path = os.path.join(pred_dir, file)
            img = image.load_img(img_path, target_size=self.img_dims)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            result = model.predict(np.vstack([x]))


            im = cv.imread(img_path)
            resized = cv.resize(im, (250, 250), interpolation=cv.INTER_LINEAR)
            axes[idx].imshow(cv.cvtColor(resized, cv.COLOR_BGR2RGB))

            label = 'CAT' if result == 0 else 'DOG' if result == 1 else 'ERR'
            axes[idx].set_title(f'{idx}: {label}')

        plt.show()

    @log_step
    def run(self):
        self.clean_slate()
        self.load_images()
        self.remove_corrupted_imgs(primary=True)
        self.process_images()
        self.gen_datasets()
        self.remove_corrupted_imgs(primary=False)
        self.train_model()


