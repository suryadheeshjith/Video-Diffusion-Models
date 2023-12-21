import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class AtariDataset(Dataset):
    def __init__(self, folder_path, total_frames, transforms=None, episode_end_frames=3):
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
        
        num_files = self.get_num_files()
        # iterate over data files
        for i in range(num_files):
            self.raw_frame_data.append(np.load(os.path.join(self.folder_path, "{}_frames.npy".format(i+1))))
            self.raw_encoding_data.append(np.load(os.path.join(self.folder_path, "{}_encodings.npy".format(i+1))))

            if len(self.raw_frame_data[-1]) != len(self.raw_encoding_data[-1]):
                min_len = min(len(self.raw_frame_data[-1]), len(self.raw_encoding_data[-1]))
                self.raw_frame_data[-1] = self.raw_frame_data[-1][:min_len]
                self.raw_encoding_data[-1] = self.raw_encoding_data[-1][:min_len]
        
        assert len(self.raw_frame_data) == len(self.raw_encoding_data)
        
        for i in range(len(self.raw_frame_data)):
            for j in range(len(self.raw_frame_data[i])-total_frames+episode_end_frames):
                total_len = len(self.raw_frame_data[i])
                
                if j > total_len-total_frames:
                    self.frames.append(np.concatenate((self.raw_frame_data[i][j:][:],self.raw_frame_data[i][:total_frames-(total_len-j)][:])))
                    self.encodings.append(np.concatenate((self.raw_encoding_data[i][j:][:],self.raw_encoding_data[i][:total_frames-(total_len-j)][:])))
                else:
                    self.frames.append(self.raw_frame_data[i][j:j+total_frames][:])
                    self.encodings.append(self.raw_encoding_data[i][j:j+total_frames][:])
                
    def get_num_files(self):
        file_pattern = "_decoded_frames.npy"
        
        if self.file_exists(file_pattern):
            return int(len(os.listdir(self.folder_path))/3)
        else:
            return int(len(os.listdir(self.folder_path))/2)  

    def file_exists(self, file_pattern):
        for filename in os.listdir(self.folder_path):
            if file_pattern in filename:
                return True
        return False

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
        frames = torch.from_numpy(self.frames[idx]).float()
        # Normalize the frames to be between 0 and 1
        frames = frames / 255.0

        encodings = torch.from_numpy(self.encodings[idx])
        if self.transforms:
            if frames.shape[-1] == 3:
                frames = frames.permute(0, 3, 1, 2)
            frames = self.transforms(frames)
        
        return frames, encodings


class AtariDataset2(Dataset):
    def __init__(self, folder_path, total_frames, transforms=None, episode_end_frames=3):
        """
        Dataset Initialization.

        Parameters:
        folder_path (str): Path to the folder containing the dataset.
        transforms (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.folder_path = folder_path
        self.transforms = transforms

        # load dataset
        self.raw_frame1_data = []
        self.raw_frame2_data = []

        self.frames1 = []
        self.frames2 = []
        
        num_files = self.get_num_files()
        # iterate over data files
        for i in range(num_files):
            self.raw_frame1_data.append(np.load(os.path.join(self.folder_path, "{}_frames.npy".format(i+1))))
            self.raw_frame2_data.append(np.load(os.path.join(self.folder_path, "{}_decoded_frames.npy".format(i+1))))

            if len(self.raw_frame1_data[-1]) != len(self.raw_frame2_data[-1]):
                min_len = min(len(self.raw_frame1_data[-1]), len(self.raw_frame2_data[-1]))
                self.raw_frame1_data[-1] = self.raw_frame1_data[-1][:min_len]
                self.raw_frame2_data[-1] = self.raw_frame2_data[-1][:min_len]
        
        assert len(self.raw_frame1_data) == len(self.raw_frame2_data)
        
        for i in range(len(self.raw_frame1_data)):
            for j in range(len(self.raw_frame1_data[i])-total_frames+episode_end_frames):
                total_len = len(self.raw_frame1_data[i])
                
                if j > total_len-total_frames:
                    self.frames1.append(np.concatenate((self.raw_frame1_data[i][j:][:],self.raw_frame1_data[i][:total_frames-(total_len-j)][:])))
                    self.frames2.append(np.concatenate((self.raw_frame2_data[i][j:][:],self.raw_frame2_data[i][:total_frames-(total_len-j)][:])))
                else:
                    self.frames1.append(self.raw_frame1_data[i][j:j+total_frames][:])
                    self.frames2.append(self.raw_frame2_data[i][j:j+total_frames][:])
                
    def get_num_files(self):
        file_pattern = "_decoded_frames.npy"
        
        if self.file_exists(file_pattern):
            return int(len(os.listdir(self.folder_path))/3)
        else:
            return int(len(os.listdir(self.folder_path))/2)  

    def file_exists(self, file_pattern):
        for filename in os.listdir(self.folder_path):
            if file_pattern in filename:
                return True
        return False

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.frames1)

    def __getitem__(self, idx):
        """
        Returns sample.

        Parameters:
        idx (int): Index of the sample to fetch.
        """
        # Fetch the data sample
        frames1 = torch.from_numpy(self.frames1[idx]).float()
        # Normalize the frames to be between 0 and 1
        frames1 = frames1 / 255.0

        frames2 = torch.from_numpy(self.frames2[idx]).float()
        frames2 = frames2 / 255.0
        if self.transforms:
            if frames1.shape[-1] == 3:
                frames1 = frames1.permute(0, 3, 1, 2)
            frames1 = self.transforms(frames1)

            if frames2.shape[-1] == 3:
                frames2 = frames2.permute(0, 3, 1, 2)
            frames2 = self.transforms(frames2)
        
        return frames1, frames2
    
def get_atari_transform(size):
    transform = transforms.Compose([
                transforms.Resize((size, size)),
                ])
    return transform



if __name__ == "__main__":
    # dataset = AtariDataset("/vast/pt2310/BreakoutNoFrameskip-v4/train", 9)
    # print("Train Dataset Length", len(dataset))
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    # print(len(os.listdir("/vast/sd5313/data/BreakoutNoFrameskip-v4/val"))/2)
    # val_dataset = AtariDataset("/vast/sd5313/data/BreakoutNoFrameskip-v4/val", 9, 64, 0)
    # print("Val Dataset Length with ep 0 ", len(val_dataset))
    val_dataset = AtariDataset("/vast/sd5313/data/Breakout_autoregressive", 9, get_atari_transform(64), 3)
    print("Dataset Length with ep 3", len(val_dataset))
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=0)

    # for i, (frames, encodings) in enumerate(dataloader):
    #     print(frames.shape, encodings.shape)
    #     break

    for i, (frames, encodings) in enumerate(val_dataloader):
        print(frames.shape, encodings.shape)
        break
