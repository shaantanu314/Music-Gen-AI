from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
class MusicDataset(Dataset):
    def __init__(self, data , data_labels):
        self.data = data
        self.data_labels = data_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx] , self.data_labels[idx]