import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class DeepFakeDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self._cache_paths()

    def _cache_paths(self):
        self.img_paths = []
        self.labels = [] if not self.is_test else None
        for _, row in self.df.iterrows():
            img_id = row['Id']
            for ext in ['.jpg', '.png', '.jpeg']:
                path = os.path.join(self.img_dir, f"{img_id}{ext}")
                if os.path.exists(path):
                    self.img_paths.append(path)
                    if not self.is_test:
                        self.labels.append(row['target_feature'])
                    break

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image, self.df.iloc[idx]['Id']
        return image, torch.tensor(self.labels[idx], dtype=torch.float32)
