import torch, os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import logging
import open_clip
import gc
from tqdm import tqdm
import itertools

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from base.utils import instantiate_from_config, get_device 

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def load_data(config):
    exp_setting = config.get('exp_setting', 'intra-subject')
    
    if exp_setting == 'intra-subject':
        test_dataset = EEGDataset(config, mode='test')
        print('init test_dataset success')
        train_dataset = EEGDataset(config, mode='train')
        print('init train_dataset success')
        test_loader = DataLoader(test_dataset, batch_size=config['data']['test_batch_size'], shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['train_batch_size'], shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
        return train_loader, test_loader, test_loader
    
    elif exp_setting == 'inter-subject':
        subjects = config['data']['subjects']
        test_dataset = EEGDataset(config, mode='test')
        print('init test_dataset success')
        
        all_subjects = [f'sub-{i:02}' for i in range(1, 11)]
        leave_one_subjects = list(set(all_subjects) - set(subjects))
        leave_one_subjects_config = config
        leave_one_subjects_config['data']['subjects'] = leave_one_subjects
        val_dataset = EEGDataset(leave_one_subjects_config, mode='test')
        print('init val_dataset success')
        train_dataset = EEGDataset(leave_one_subjects_config, mode='train')
        print('init train_dataset success')
        test_loader = DataLoader(test_dataset, batch_size=config['data']['test_batch_size'], shuffle=False, drop_last=False, num_workers=25)
        val_loader = DataLoader(val_dataset, batch_size=config['data']['val_batch_size'], shuffle=True, drop_last=False, num_workers=32)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['train_batch_size'], shuffle=True, drop_last=False, num_workers=32)
        return train_loader, val_loader, test_loader
    
class EEGDataset(Dataset):
    def __init__(self, config, mode):
        self.config = config
        self.data_dir = config['data']['data_dir']
        self.subjects = config['data']['subjects']
        print(f'subjects:{self.subjects}')
        self.mode = mode
        self.name = config['name']
        self.model_type = config['data']['model_type']
        self.selected_ch = config['data']['selected_ch']
        self.channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                        'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
                        'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                        'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
                        'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                        'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                        'O1', 'Oz', 'O2']
        
        if self.selected_ch == "ALL":
            self.selected_ch = self.channels
        elif self.selected_ch == "O": # Occipital
            self.selected_ch = ['PO7', 'PO3', 'POz', 'PO4', 'PO8','O1', 'Oz', 'O2']
        elif self.selected_ch == "P": # Parietal
            self.selected_ch = ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8']
        elif self.selected_ch == "C":  # Central
            self.selected_ch = ['C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 
                        'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']
        elif self.selected_ch == "T":  # Temporal
            self.selected_ch = ['FT9', 'FT7', 'FT8', 'FT10', 'TP9', 'TP7', 'TP8', 'TP10', 'T7', 'T8']
        elif self.selected_ch == "F":  # Frontal
            self.selected_ch = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
                        'F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6', 'F8', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6']
        elif self.selected_ch == "OP": # Occipital + Parietal
            self.selected_ch = ['P7', 'P5', 'P3', 'P1','Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8','O1', 'Oz', 'O2']
        elif self.selected_ch == "OPC":  # Occipital + Parietal + Central
            self.selected_ch = ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                        'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2',
                        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                        'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']
        elif self.selected_ch == "OPT":
            self.selected_ch = ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                                'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2',
                                'FT9', 'FT7', 'FT8', 'FT10', 'TP9', 'TP7', 'TP8', 'TP10',
                                'T7', 'T8']
        elif self.selected_ch == "OPCT":
            self.selected_ch = ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                                'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2',
                                'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                                'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
                                'FT9', 'FT7', 'FT8', 'FT10', 'TP9', 'TP7', 'TP8', 'TP10', 'T7', 'T8']

        self.avg = config['data'][f"{mode}_avg"]
        self.timesteps = config['data']['timesteps']

        self.n_cls = 1654 if self.mode == 'train' else 200
        self.per_trials = 4 if self.mode == 'train' else 80

        self.data_paths = [os.path.join(self.data_dir, subject, f'{mode}.pt') for subject in self.subjects]
        self.loaded_data = [self.load_data(data_path) for data_path in self.data_paths]
        
        self.trial_subject = self.loaded_data[0]['eeg'].shape[0]
        self.trial_all_subjects = self.trial_subject * len(self.subjects)

        # Feature paths
        feature_dir = self.config['data']['feature_dir']
        
        high_level_filename = f"high_level_{mode}.pt"
        low_level_filename  = f"low_level_{mode}.pt"

        high_level_path = os.path.join(feature_dir, high_level_filename)
        low_level_path  = os.path.join(feature_dir, low_level_filename)

        print("\n========== [EEGDataset Feature Paths] ==========")
        print(f"[Mode]            : {mode}")
        print(f"[High Level Path] : {high_level_path}")
        print(f"[Low Level Path]  : {low_level_path}")
        print("================================================\n")
        
        self.high_level_feature = torch.load(high_level_path, map_location="cpu")
        self.low_level_feature = torch.load(low_level_path, map_location="cpu")

    def load_data(self, data_path):
        logging.info(f"----load {data_path.rsplit('1000HZ', 1)[-1]}----")
        loaded_data = torch.load(data_path, weights_only=False)
        loaded_data['eeg'] = torch.as_tensor(loaded_data['eeg'], dtype=torch.float32)

        if self.selected_ch:
            selected_idx = [self.channels.index(ch) for ch in self.selected_ch]
            loaded_data['eeg'] = loaded_data['eeg'][:, :, selected_idx]
        if self.avg:
            avg_data = {}
            avg_data['eeg'] = loaded_data['eeg'].mean(axis=1)
            avg_data['label'] = loaded_data['label'][:, 0]
            avg_data['img'] = loaded_data['img'][:, 0]
            avg_data['text'] = loaded_data['text'][:, 0]
            avg_data['session'] = loaded_data['session']
            avg_data['times'] = loaded_data['times']
            loaded_data = avg_data
        else:
            _data = {}
            _data['eeg'] = loaded_data['eeg'].reshape(-1, *loaded_data['eeg'].shape[2:])
            _data['eeg_avg'] = loaded_data['eeg'].mean(axis=1)
            _data['label'] = loaded_data['label'].reshape(-1)
            _data['img'] = loaded_data['img'].reshape(-1)
            _data['text'] = loaded_data['text'].reshape(-1)
            _data['session'] = loaded_data['session'].reshape(-1)
            _data['times'] = loaded_data['times']
            loaded_data = _data
        
        for k, v in loaded_data.items():
            if k in ['eeg', 'label', 'img', 'text', 'session']:
                logging.info(f"{k}: {v.shape}")
        return loaded_data    
    
    def __getitem__(self, index):
        subject = index // self.trial_subject
        trial_index = index % self.trial_subject

        eeg = self.loaded_data[subject]['eeg'][trial_index].float()
        if self.avg:
            eeg_mean = eeg
        else:
            eeg_mean = self.loaded_data[subject]['eeg_avg'][trial_index // self.per_trials].float()

        label = self.loaded_data[subject]['label'][trial_index]
        img_path = self.loaded_data[subject]['img'][trial_index]

        # Use renamed attributes
        high_level_feature = self.high_level_feature[trial_index]
        low_level_feature = self.low_level_feature[trial_index]

        session = self.loaded_data[subject]['session'][trial_index]

        sample = {
            'idx': torch.tensor(index, dtype=torch.long),
            'trial_index': torch.tensor(trial_index, dtype=torch.long),
            'eeg': eeg[:, self.timesteps[0]:self.timesteps[1]],
            'label': torch.tensor(label, dtype=torch.long),
            'img_path': img_path,  
            # Renamed keys
            'high_level_feature': high_level_feature,
            'low_level_feature': low_level_feature,
            'session': torch.tensor(session, dtype=torch.long),
            'subject': torch.tensor(subject, dtype=torch.long),
            'eeg_mean': eeg_mean[:, self.timesteps[0]:self.timesteps[1]],
        }

        return sample
    
    def __len__(self):
        return self.trial_all_subjects