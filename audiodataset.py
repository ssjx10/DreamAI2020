import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import glob
import numpy as np
from util import read_filepaths
from PIL import Image
import torchaudio
import torchaudio.transforms as AT
from torchvision import transforms
from sklearn.model_selection import train_test_split
import librosa
import json

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

# audio augmentation
# add the noise
def addWhiteNoise(audio):
    wn = np.random.randn(len(audio))
    audio_wn = audio + 0.005*wn
    return audio_wn

# Shifting the sound
def shift(audio):
    audio_roll = np.roll(audio, 1600)
    return audio_roll

# stretching the sound
def stretch(audio, rate=1):
    input_length = 16000*4
    rate = np.random.randint(75, 131)*0.01
    audio = librosa.effects.time_stretch(audio, rate)
    if len(audio)>input_length:
        audio = audio[:input_length]
    else:
        audio = np.pad(audio, (0, max(0, input_length - len(audio))), "constant")

    return audio

class ConcatDataset_pair(Dataset):
    def __init__(self, aX, ay, iX, iy, mode):
        
        a_labels = np.where(ay == 'healthy', 0, 1)
        i_labels = np.where(iy == 'NON-COVID-19', 0, 1)
        
        n_sample0 = np.sum(a_labels == 0)
        n_sample1 = np.sum(a_labels == 1)
        audio_idx0 = aX[a_labels == 0]
        audio_idx0_labels = a_labels[a_labels == 0]
        audio_idx1 = aX[a_labels == 1]
        audio_idx1_labels = a_labels[a_labels == 1]
        
        # extract a image subsample(n == audio sample)
        idx0, = np.where(i_labels == 0)
        idx1, = np.where(i_labels == 1)
        idx0_sample = np.random.choice(idx0, n_sample0)
        idx1_sample = np.random.choice(idx1, n_sample1)
        new_i_idx = np.concatenate([idx0_sample, idx1_sample])
        
        # audio_image pair
        new_audio = np.concatenate([audio_idx0, audio_idx1])
        new_a_labels = np.concatenate([audio_idx0_labels, audio_idx1_labels])
        new_image = iX[new_i_idx]
        new_i_labels =  i_labels[new_i_idx]
        
        # shuffle
        indices = np.arange(new_audio.shape[0])
        np.random.shuffle(indices)

        new_audio = new_audio[indices]
        new_a_labels = new_a_labels[indices]
        new_image = new_image[indices]
        new_i_labels = new_i_labels[indices]
        
        self.aX = new_audio
        self.ay = new_a_labels
        self.iX = new_image
        self.iy = new_i_labels
        self.mode = mode
        
        _, cnts = np.unique(self.ay, return_counts=True)
        print("{} audio examples =  {}".format(mode, len(self.ay)), cnts)
        _, cnts = np.unique(self.iy, return_counts=True)
        print("{} image examples =  {}".format(mode, len(self.iy)), cnts)
        
        if (mode == 'train'):
            self.transform = train_transformer
        elif mode == 'test' or mode == 'valid':
            self.transform = val_transformer
        
        if mode == 'train':
            Training_pp=[]
            Training_nn=[]
            Training_pn=[]
            Training_np=[]
            for i_tr in range(len(new_i_labels)):
                for a_tr in range(len(new_a_labels)):
                    if a_labels[a_tr] == new_i_labels[i_tr]:
                        if a_labels[a_tr] == 1:
                            Training_pp.append([a_tr,i_tr, 1])
                        elif a_labels[a_tr] == 0:
                            Training_nn.append([a_tr,i_tr, 1])
                    else:
                        if a_labels[a_tr] == 1:
                            Training_np.append([a_tr,i_tr, 0])
                        elif a_labels[a_tr] == 0:
                            Training_pn.append([a_tr,i_tr, 0])
            print("Class P : ", len(Training_pp) + len(Training_nn), " N : ", len(Training_np) + len(Training_pn))

            self.Training_pp = Training_pp
            self.Training_nn = Training_nn
            self.Training_pn = Training_pn
            self.Training_np = Training_np

            random.shuffle(self.Training_nn)
            random.shuffle(self.Training_np)
            random.shuffle(self.Training_pn)
            random.shuffle(self.Training_pp)
            self.datas = self.Training_nn[:len(Training_pp)//2] + self.Training_pp[:len(Training_pp)//2] + self.Training_np[:len(Training_pp)//2] + self.Training_pn[:len(Training_pp)//2]
            random.shuffle(self.datas)
        else:
            self.datas = new_a_labels

    def __getitem__(self, i):
        if self.mode == 'train': 
            a_idx, i_idx, domain = self.datas[i]

            mel = torch.from_numpy(self.aX[a_idx])
            a_label_tensor = torch.tensor(self.ay[a_idx], dtype=torch.long)
            image_tensor = self.transform(self.iX[i_idx])
            i_label_tensor = torch.tensor(self.iy[i_idx], dtype=torch.long)
            p_image = self.transform(self.iX[a_idx])

            return mel, a_label_tensor, image_tensor, i_label_tensor, p_image
        
        else:
            mel = self.aX[i]
            mel = torch.from_numpy(mel)
            label_tensor = torch.tensor(self.ay[i], dtype=torch.long)
            p_image = self.transform(self.iX[i])
            
            return mel, p_image, label_tensor

    def __len__(self):
        return len(self.datas)

class ConcatDataset(Dataset):
    def __init__(self, aX, ay, iX, iy, mode):
        
        a_labels = np.where(ay == 'healthy', 0, 1)
        i_labels = np.where(iy == 'NON-COVID-19', 0, 1)
        
        self.segment_length = 16000*2
        self.aX = aX
        self.ay = a_labels
        
        n_sample0 = np.sum(a_labels == 0)
        n_sample1 = np.sum(a_labels == 1)
        idx0, = np.where(i_labels == 0)
        idx1, = np.where(i_labels == 1)
        idx0_sample = np.random.choice(idx0, n_sample0)
        idx1_sample = np.random.choice(idx1, n_sample1)
        new_i_idx = np.concatenate([idx0_sample, idx1_sample])
        
        self.iX = iX
        new_i_labels =  i_labels
        self.iy = new_i_labels
        
        _, cnts = np.unique(self.ay, return_counts=True)
        print("{} audio examples =  {}".format(mode, len(self.ay)), cnts)
        _, cnts = np.unique(self.iy, return_counts=True)
        print("{} image examples =  {}".format(mode, len(self.iy)), cnts)
        
        if (mode == 'train'):
            self.transform = train_transformer
        elif mode == 'test' or mode == 'valid':
            self.transform = val_transformer
        
        Training_pp=[] # pp -> image&audio
        Training_nn=[]
        Training_pn=[]
        Training_np=[]
        for i_tr in range(len(new_i_labels)):
            for a_tr in range(len(a_labels)):
                if a_labels[a_tr] == new_i_labels[i_tr]:
                    if a_labels[a_tr] == 1:
                        Training_pp.append([a_tr,i_tr, 1])
                    elif a_labels[a_tr] == 0:
                        Training_nn.append([a_tr,i_tr, 1])
                else:
                    if a_labels[a_tr] == 1:
                        Training_np.append([a_tr,i_tr, 0])
                    elif a_labels[a_tr] == 0:
                        Training_pn.append([a_tr,i_tr, 0])
        print("Class P : ", len(Training_pp) + len(Training_nn), " N : ", len(Training_np) + len(Training_pn))
        
        self.Training_pp = Training_pp
        self.Training_nn = Training_nn
        self.Training_pn = Training_pn
        self.Training_np = Training_np
        
        random.shuffle(self.Training_nn)
        random.shuffle(self.Training_np)
        random.shuffle(self.Training_pn)
        random.shuffle(self.Training_pp)
        self.datas = self.Training_nn[:len(Training_pp)//8] + self.Training_pp[:len(Training_pp)//8] + self.Training_np[:len(Training_pp)//8] + self.Training_pn[:len(Training_pp)//8]
        random.shuffle(self.datas)
        

    def __getitem__(self, i):
        a_idx, i_idx, domain = self.datas[i]
        
        audio = self.aX[a_idx]
        # Take segment
        if len(audio) >= self.segment_length:
            audio = seg2(audio, self.segment_length)
        else:
            audio = np.pad(audio, (0, self.segment_length - len(audio)), "constant")
#         audio = addWhiteNoise(audio)
        audio = shift(audio)
#         audio = stretch(audio)
        audio = torch.from_numpy(audio)
        audio = audio.float()
        
        audio = audio.unsqueeze(0)
        mel = mel_spectrogram(audio)
        a_label_tensor = torch.tensor(self.ay[a_idx], dtype=torch.long)
        image_tensor = self.transform(self.iX[i_idx])
        i_label_tensor = torch.tensor(self.iy[i_idx], dtype=torch.long)

        return mel, a_label_tensor, image_tensor, i_label_tensor

    def __len__(self):
        return len(self.datas)

class ConcatDataset2(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        
        a_labels = self.datasets[0].labels
        a_labels = np.where(a_labels == 'healthy', 0, 1)
        i_labels = self.datasets[1].labels
        i_labels = np.where(i_labels == 'NON-COVID-19', 0, 1)
        
        n_sample0 = np.sum(a_labels == 0)
        n_sample1 = np.sum(a_labels == 1)
        idx0, = np.where(i_labels == 0)
        idx1, = np.where(i_labels == 1)
        idx0_sample = np.random.choice(idx0, n_sample0)
        idx1_sample = np.random.choice(idx1, n_sample1)
        new_i_idx = np.concatenate([idx0_sample, idx1_sample])
        
        Training_P=[]
        Training_N=[]
        for a_tr in range(len(a_labels)):
            for i_tr in new_i_idx:
                if a_labels[a_tr] == i_labels[i_tr]:
                    Training_P.append([a_tr,i_tr, 1])
                else:
                    Training_N.append([a_tr,i_tr, 0])
        print("Class P : ", len(Training_P), " N : ", len(Training_N))
        
        random.shuffle(Training_N)
        random.shuffle(Training_P)
        self.datas = Training_P + Training_N
        random.shuffle(self.datas)

    def __getitem__(self, i):
        random.shuffle(self.datas)
        a_idx, i_idx, domain = self.datas[i]

        return self.datasets[0][a_idx], self.datasets[1][i_idx]

    def __len__(self):
        return len(self.datas) // 10
    
# +++++++++++++++ AUDIO +++++++++++++++++++++++
mel_spectrogram = nn.Sequential(
            AT.MelSpectrogram(sample_rate=16000, 
                              n_fft=512, 
                              win_length=400,
                              hop_length=160,
                              n_mels=80,
                              f_max=8000
                              ),
            AT.AmplitudeToDB()
)

def seg1(audio, segment_length):
    max_audio_start = len(audio) - segment_length
    audio_start = random.randint(0, max_audio_start)
    audio_s = audio[audio_start:audio_start + segment_length]
    return audio_s

def seg2(audio, segment_length):
    max_audio_start = len(audio) - segment_length
    audio_start = np.argmax(audio) - segment_length//2
    if audio_start < 0:
        audio_start = 0
    if audio_start > max_audio_start:
        audio_start = max_audio_start
    audio_s = audio[audio_start:audio_start + segment_length]
    return audio_s

class CoswaraDataset3(Dataset):
    """
    Code for reading the CoswaraDataset
    """

    def __init__(self, x, labels, mode):
        

        self.AudioDICT = {'healthy': 0, 'positive': 1}
        self.x = x
        self.labels = labels
        
        _, cnts = np.unique(np.where(self.labels == 'healthy', 0, 1), return_counts=True)
        print("{} examples =  {}".format(mode, len(self.labels)), cnts)
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        mel = self.x[index]
        mel = torch.from_numpy(mel)
    
        label_tensor = torch.tensor(self.AudioDICT[self.labels[index]], dtype=torch.long)

        return mel, label_tensor

class CoswaraDataset2(Dataset):
    """
    Code for reading the CoswaraDataset
    """

    def __init__(self, x, labels, mode, segment_length=16000):
        

        self.AudioDICT = {'healthy': 0, 'positive': 1}
        self.segment_length = segment_length
        self.x = x
        self.labels = labels
        
        _, cnts = np.unique(np.where(self.labels == 'healthy', 0, 1), return_counts=True)
        print("{} examples =  {}".format(mode, len(self.labels)), cnts)
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        audio = self.x[index]
        audio = torch.from_numpy(audio)
    
        # Take segment
        if audio.size(0) >= self.segment_length:
            audio = seg2(audio, self.segment_length)
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data

        audio = audio.unsqueeze(0)
        mel = mel_spectrogram(audio)
        label_tensor = torch.tensor(self.AudioDICT[self.labels[index]], dtype=torch.long)

        return mel, label_tensor


class CoswaraDataset(Dataset):
    """
    Code for reading the CoswaraDataset
    """

    def __init__(self, mode, n_classes=2, dataset_path='../datasets/data/coswara/*/*/', segment_length=16000):
        

        self.CLASSES = n_classes
        self.AudioDICT = {'healthy': 0, 'positive': 1}
        self.segment_length = segment_length
        pid_list = glob.glob(dataset_path)
        
        paths = []
        labels = []
        for pid in pid_list:
            json_file = pid + 'metadata.json'
            with open(json_file) as json_file:
                json_data = json.load(json_file)
                status = json_data["covid_status"]
            if status == 'positive_mild' or status == 'positive_moderate':
                status = 'positive'
            if status != 'healthy' and status != 'positive':
                continue
            file_list = glob.glob(pid + '*.wav')
            for f in file_list:
                if 'cough' not in f: #and 'breathing' not in f:
                    continue
                paths.append(f)
                labels.append(status)
        paths = np.array(paths)
        labels = np.array(labels)
        
        n_sample = np.sum(labels == 'positive')
        h_paths = paths[labels == 'healthy']
        h_labels = labels[labels == 'healthy']
        idx_sample = np.random.choice(len(h_paths), n_sample)
        new_paths = np.concatenate([h_paths[idx_sample], paths[labels == 'positive']])
        new_labels = np.concatenate([h_labels[idx_sample], labels[labels == 'positive']])
        x, x_test, y, y_test = train_test_split(new_paths, new_labels, test_size=0.2, shuffle=True, stratify=new_labels, random_state=10)
        
        if (mode == 'train' or mode == 'valid'):
            x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.1, shuffle=True, stratify=y, random_state=10)
            if mode == 'train':
                self.paths, self.labels = x, y
            elif mode == 'valid':
                self.paths, self.labels = x_valid, y_valid
        elif (mode == 'test'):
            self.paths, self.labels = x_test, y_test
        _, cnts = np.unique(self.labels, return_counts=True)
        print("{} examples =  {}".format(mode, len(self.paths)), cnts)
        
        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        audio = self.load_audio(self.paths[index])
        audio = torch.from_numpy(audio)
    
        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment_length]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_length - audio.size(0)), 'constant').data

        audio = audio.unsqueeze(0)
        label_tensor = torch.tensor(self.AudioDICT[self.labels[index]], dtype=torch.long)

        return audio, label_tensor

    def load_audio(self, path):
        if not os.path.exists(path):
            print("AUDIO DOES NOT EXIST {}".format(path))
        audio, sr = librosa.load(path, sr=16000)
#         image_tensor = self.transform(image)

        return audio





