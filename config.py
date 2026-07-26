```python
import torch

class Config:
    # Data paths 
    train_csv = "/kaggle/input/datasets/dayman2006/images/dataset/train_solution.csv"  
    train_img_dir = "/kaggle/input/datasets/dayman2006/images/dataset/train_images"     
    test_img_dir = "/kaggle/input/datasets/dayman2006/images/dataset/test_images"      

    # Model
    img_size = 224
    batch_size = 64
    epochs = 20
    lr = 1e-4
    weight_decay = 1e-4
    pos_weight = 2.5      # для балансировки классов 
    threshold = 0.45      # порог классификации
    dropout = 0.3

    # Training
    use_amp = True
    grad_clip = 1.0
    patience = 5
    seed = 42

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
