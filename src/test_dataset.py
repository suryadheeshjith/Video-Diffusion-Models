import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class AtariDataset(Dataset):
    def __init__(self, folder_path, transforms=None):
        """
        Dataset Initialization.

        Parameters:
        folder_path (str): Path to the folder containing the dataset.
        transforms (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.folder_path = folder_path
        self.transforms = transforms

        # load dataset
        self.raw_frame_data = []
        self.raw_encoding_data = []

        self.frames = []
        self.encodings = []
        # iterate over data files
        for i in range(int(len(os.listdir(self.folder_path))/2)):
        # for i in range(1):
            self.raw_frame_data.append(np.load(os.path.join(self.folder_path, "{}_frames.npy".format(i+1))))
            self.raw_encoding_data.append(np.load(os.path.join(self.folder_path, "{}_encodings.npy".format(i+1))))

            assert len(self.raw_frame_data[-1]) == len(self.raw_encoding_data[-1])
        
        assert len(self.raw_frame_data) == len(self.raw_encoding_data)
        
        for i in range(len(self.raw_frame_data)):
            for j in range(len(self.raw_frame_data[i])-9):
                self.frames.append(self.raw_frame_data[i][j:j+9][:])
                self.encodings.append(self.raw_encoding_data[i][j:j+9][:])
                

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.frames)

    def __getitem__(self, idx):
        """
        Returns sample.

        Parameters:
        idx (int): Index of the sample to fetch.
        """
        # Fetch the data sample
        frames = torch.from_numpy(self.frames[idx])
        encodings = torch.from_numpy(self.encodings[idx])

        return frames, encodings

dataset = AtariDataset("./saved_npy/BreakoutNoFrameskip-v4/train")
print("Train Dataset Length", len(dataset))
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

val_dataset = AtariDataset("./saved_npy/BreakoutNoFrameskip-v4/val")
print("Val Dataset Length", len(val_dataset))
val_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

for i, (frames, encodings) in enumerate(dataloader):
    print(frames.shape, encodings.shape)
    break

for i, (frames, encodings) in enumerate(val_dataloader):
    print(frames.shape, encodings.shape)
    break