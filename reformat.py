from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import os

class Formatter:
    def __init__(self, SIZE, FORMAT, DIR):
        self.IMG_SIZE = SIZE
        self.IMG_FORMAT = FORMAT
        self.IMG_DIR = DIR

    def resize(self, src_dir, dest_dir):
        for idx, filename in enumerate(os.listdir(src_dir)):
            try:
                img = Image.open(os.path.join(src_dir,filename))
                img = img.resize((self.IMG_SIZE[0], self.IMG_SIZE[1]), Image.ANTIALIAS)
                img.save(os.path.join(dest_dir,filename))
                print(f"Image {idx} Done")
                os.remove(os.path.join(src_dir, filename))
            except:
                print(f"ERROR WITH {filename}")
                os.remove(f"{src_dir}/{filename}")

    def batch_rename(self, newLabel):
        for idx, filename in enumerate(os.listdir(self.IMG_DIR)):
            try:
                os.rename(f"{self.IMG_DIR}/{filename}", f"{self.IMG_DIR}/{newLabel}_{idx}.{self.IMG_FORMAT}")
                print(f"Image {idx} renamed")
            except:
                print(f"ERROR WITH {filename}")
                os.remove(os.path.join(self.IMG_DIR, filename))

    def batch_extension_change(self):
        for idx,filename in enumerate(os.listdir(self.IMG_DIR)):
            try:
                label, suffix = filename.split('.')
                new_suffix = self.IMG_FORMAT
                new_filename = label + '.' + new_suffix
                os.rename(self.IMG_DIR + '/' + filename, self.IMG_DIR + '/' + new_filename)
                print(f"Image {idx} done")
            except:
                print(f"ERROR WITH {filename}")
        