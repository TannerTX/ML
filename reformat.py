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
                img = Image.open(os.path.join(src_dir, filename))
                img = img.resize((self.IMG_SIZE[0], self.IMG_SIZE[1]), Image.Resampling.LANCZOS)
                img.save(os.path.join(dest_dir, filename))
                if idx % 1000 == 0:
                    print(f"[Resize] Processed {idx} images...")
                os.remove(os.path.join(src_dir, filename))
            except Exception as e:
                print(f"[Resize] ERROR with {filename}: {e}")
                os.remove(os.path.join(src_dir, filename))

    def batch_rename(self, newLabel):
        for idx, filename in enumerate(os.listdir(self.IMG_DIR)):
            try:
                new_name = f"{newLabel}_{idx}.{self.IMG_FORMAT}"
                os.rename(os.path.join(self.IMG_DIR, filename), os.path.join(self.IMG_DIR, new_name))
                if idx % 1000 == 0:
                    print(f"[Rename] Renamed {idx} images...")
            except Exception as e:
                print(f"[Rename] ERROR with {filename}: {e}")
                os.remove(os.path.join(self.IMG_DIR, filename))

    def batch_extension_change(self):
        for idx, filename in enumerate(os.listdir(self.IMG_DIR)):
            try:
                label, _ = os.path.splitext(filename)
                new_filename = f"{label}.{self.IMG_FORMAT}"
                os.rename(os.path.join(self.IMG_DIR, filename), os.path.join(self.IMG_DIR, new_filename))
                if idx % 1000 == 0:
                    print(f"[ExtChange] Updated {idx} extensions...")
            except Exception as e:
                print(f"[ExtChange] ERROR with {filename}: {e}")
