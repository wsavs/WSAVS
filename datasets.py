import os
import csv
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from scipy import signal
import random
import json
import pickle
import xml.etree.ElementTree as ET
from audio_io import load_audio_av, open_audio_av
import torchaudio
from torchaudio import transforms as aT
import torch

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def load_image(path):
    return Image.open(path).convert('RGB')


def load_waveform(path, dur=3.):
    # Load audio
    print('path:', path)
    audio_ctr = open_audio_av(path)
    print('audio_ctr:', audio_ctr)
    audio_dur = audio_ctr.streams.audio[0].duration * audio_ctr.streams.audio[0].time_base
    audio_ss = max(float(audio_dur)/2 - dur/2, 0)
    print('audio_ss:', audio_ss)
    audio, samplerate = load_audio_av(container=audio_ctr, start_time=audio_ss, duration=dur)

    # To Mono
    audio = np.clip(audio, -1., 1.).mean(0)

    # Repeat if audio is too short
    if audio.shape[0] < samplerate * dur:
        n = int(samplerate * dur / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    waveform = audio[:int(samplerate * dur)]

    return waveform, samplerate

def log_mel_spectrogram(waveform, samplerate):
    frequencies, times, spectrogram = signal.spectrogram(waveform, samplerate, nperseg=512, noverlap=274)
    spectrogram = np.log(spectrogram + 1e-7)
    return spectrogram

def load_audio_torch(path, dur=3., fps=None):
    # Load audio
    si, _ = torchaudio.info(str(path))
    info = {}
    info['samplerate'] = si.rate
    info['samples'] = si.length // si.channels

    if info['samples'] > 0:
        audio_dur = info['samples'] / si.rate
    else:
        audio_stream = open_audio_av(path).streams.audio[0]
        audio_dur = audio_stream.duration * audio_stream.time_base

    offset = int(max(float(audio_dur)/2 - dur/2, 0) * si.rate)
    num_frames = int(dur * si.rate)
    audio, samplerate = torchaudio.load(path, offset=offset, num_frames=num_frames)

    # Resample
    if fps is not None and fps != samplerate:
        # audio = torchaudio.functional.resampler(audio, samplerate, fps)
        resampler = aT.Resample(samplerate, fps)
        audio = resampler(audio)
        samplerate = fps
        num_frames = int(dur * fps)

    # To Mono
    audio = np.clip(audio, -1., 1.).mean(0)

    # Repeat if audio is too short
    if audio.shape[0] < num_frames:
        n = int(num_frames / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    audio = audio[:num_frames]

    return audio, samplerate

def log_mel_spectrogram(waveform, sample_rate):
    hop_len = int(sample_rate * 0.0234)
    n_fft = np.power(2, int(np.log2(sample_rate * 0.05))+1)
    if type(waveform) != torch.Tensor:
        waveform = torch.from_numpy(waveform)
    mel = aT.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_len)(waveform)
    if mel.sum() == 0.:
        raise RuntimeError("Silent video")
    log_mel = torch.log10(mel + 1e-7)[None]
    return log_mel

def load_all_masks(annotation_dir, format='avsbench'):
    gt_masks = {}
    if format in {'avsbench', 'vgginstruments'}:
        anno_files = os.listdir(annotation_dir)
        for filename in anno_files:
            file = filename.split('.png')[0]
            gt_masks[file] = f"{annotation_dir}/{filename}"
    return gt_masks

def mask2gtmap(gt_mask_path, format='avsbench'):
    if format in {'avsbench'}:
        gt_map = np.asarray(Image.open(gt_mask_path).convert('1')).astype(int)
    elif format in {'vgginstruments'}:
        with open(gt_mask_path, 'rb') as f:
            gt_mask = pickle.load(f)
        gt_map = cv2.resize(gt_mask, (224,224), interpolation=cv2.INTER_NEAREST)
    return gt_map


class AudioVisualDataset(Dataset):
    def __init__(self, image_files, audio_files, image_path, audio_path, mode='train', 
            audio_dur=3., image_transform=None, audio_transform=None, all_masks=None, mask_format='avsbench', 
            num_classes=0, class_labels=None):
        super().__init__()
        self.audio_path = audio_path
        self.image_path = image_path

        self.mode = mode
        self.audio_dur = audio_dur
        self.audio_sample_rate = 8000

        self.audio_files = audio_files
        self.image_files = image_files
        self.all_masks = all_masks
        self.mask_format = mask_format
        self.class_labels = class_labels
        self.num_classes = num_classes

        self.image_transform = image_transform
        self.audio_transform = audio_transform

    def getitem(self, idx):

        image_path = self.image_path
        audio_path = self.audio_path

        anno = {}
        if self.all_masks is not None:
            gt_mask_path = self.all_masks[idx]
            anno['gt_map'] = mask2gtmap(gt_mask_path, format=self.mask_format)
            anno['gt_mask'] = 1             # 1 for samples w. gt_map
            bb = torch.ones((1, 4)).long()
            anno['bboxes'] = bb
            
        # class_label = torch.zeros(self.num_classes)
        # class_idx = self.class_labels[idx]
        # class_label[class_idx] = 1
        if self.class_labels is not None:
            anno['class'] = torch.Tensor([self.class_labels[idx]])

        file = self.image_files[idx]
        file_id = file.split('.')[0]

        # Image
        img_fn = image_path + self.image_files[idx]
        frame = self.image_transform(load_image(img_fn))

        # # Audio
        audio_fn = audio_path + self.audio_files[idx]
        # waveform, samplerate = load_waveform(audio_fn)
        # spectrogram = self.audio_transform(log_mel_spectrogram(waveform, samplerate))
        waveform, sample_rate = load_audio_torch(audio_fn, dur=self.audio_dur, fps=self.audio_sample_rate)
        # print('waveform:', waveform.shape)
        # print('sample_rate:', sample_rate)
        log_mel = log_mel_spectrogram(waveform, sample_rate)
        # print('log_mel:', log_mel.shape)

        return frame, log_mel, anno, file_id

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception:
            return self.getitem(random.sample(range(len(self)), 1)[0])


def get_train_dataset(args):
    audio_path = f"{args.train_data_path}/audio/"
    image_path = f"{args.train_data_path}/frames/"
    mask_format = {'avsbench': 'avsbench'}[args.testset]

    # List directory
    audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path) if fn.endswith('.wav')}
    image_files = {fn.split('.jpg')[0] for fn in os.listdir(image_path) if fn.endswith('.jpg')}
    if args.trainset in {'avsbench'}:
        avail_audio_files = []
        for image_file in image_files:
            # print('image_file:', image_file)
            if image_file[:-2] in audio_files:
                avail_audio_files.append(image_file)
        audio_files = {file for file in avail_audio_files}
    print('audio_files:', list(audio_files)[:100])
    print('image_files:', list(image_files)[:100])

    avail_files = audio_files.intersection(image_files)
    print(f"{len(avail_files)} available files")

    # Subsample if specified
    subset = set([line.split(',')[0] for line in open(f"metadata/{args.trainset}_train.txt").read().splitlines()])
    avail_files = avail_files.intersection(subset)
    print(f"{len(avail_files)} valid subset files")
    
    avail_files = sorted(list(avail_files))
    audio_files = [dt[:-2]+'.wav' for dt in avail_files]
    image_files = [dt+'.jpg' for dt in avail_files]

    # Pseudo masks
    if args.train_pseudo_gt_path is not None:
        print('mask_format:', mask_format)
        all_pseudo_masks = load_all_masks(args.train_pseudo_gt_path, format=mask_format)
        all_pseudo_masks = [all_pseudo_masks[fn.split('.jpg')[0]] for fn in image_files]
    else:
        all_pseudo_masks = None

    class_labels = []
    all_classes = sorted(set([line.split(',')[1] for line in open(f"metadata/{args.trainset}_train.txt").read().splitlines()]))
    num_classes = len(all_classes)
    fns2cls = {line.split(',')[0]:line.split(',')[1] for line in open(f"metadata/{args.trainset}_train.txt").read().splitlines()}
    for dt in avail_files:
        cls = all_classes.index(fns2cls[dt])
        class_labels.append(cls)

    print('class_labels:', class_labels[:10])
    print('all_classes:', all_classes)
    print('num_classes:', len(all_classes))

    # Transforms
    image_transform = transforms.Compose([
        transforms.Resize(int(224 * 1.1), Image.BICUBIC),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0], std=[12.0])])

    return AudioVisualDataset(
        mode='train',
        image_files=image_files,
        audio_files=audio_files,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=3.,
        image_transform=image_transform,
        audio_transform=audio_transform,
        num_classes=num_classes,
        class_labels=class_labels,
        all_masks=all_pseudo_masks,
    )


def get_val_test_dataset(args):
    audio_path = args.test_data_path + 'audio/'
    image_path = args.test_data_path + 'frames/'

    if args.testset in ['avsbench']:
        if args.mode == 'train':
            testcsv = 'metadata/avsbench_val.csv'
        elif args.mode == 'test':
            testcsv = 'metadata/avsbench_test.csv'
    else:
        raise NotImplementedError
    mask_format = {'avsbench': 'avsbench'}[args.testset]

    #  Retrieve list of audio and video files
    testset = set([item[0] for item in csv.reader(open(testcsv))])

    # Intersect with available files
    audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path)}
    image_files = {fn.split('.jpg')[0] for fn in os.listdir(image_path)}
    if args.testset in {'avsbench'}:
        avail_audio_files = []
        for image_file in image_files:
            if image_file[:-2] in audio_files:
                avail_audio_files.append(image_file)
        audio_files = {file for file in avail_audio_files}

    avail_files = audio_files.intersection(image_files)
    testset = testset.intersection(avail_files)
    print(f"{len(testset)} files for testing")

    testset = sorted(list(testset))
    image_files = [dt+'.jpg' for dt in testset]
    if args.testset in {'avsbench'}:
        audio_files = [dt[:-2]+'.wav' for dt in testset]
    else:
        audio_files = [dt+'.wav' for dt in testset]

    # Ground-truth masks
    print('mask_format:', mask_format)
    all_masks = load_all_masks(args.test_gt_path, format=mask_format)
    all_masks = [all_masks[fn.split('.jpg')[0]] for fn in image_files]

    class_labels = []
    all_classes = sorted(set([line.split(',')[1] for line in open(f"metadata/{args.trainset}_train.txt").read().splitlines()]))
    num_classes = len(all_classes)
    fns2cls = {item[0]:item[1] for item in csv.reader(open(testcsv))}
    for dt in testset:
        cls = all_classes.index(fns2cls[dt])
        class_labels.append(cls)
        
    # Transforms
    image_transform = transforms.Compose([
        transforms.Resize((224, 224), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.0], std=[12.0])])

    return AudioVisualDataset(
        mode='test',
        image_files=image_files,
        audio_files=audio_files,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=5.,
        all_masks=all_masks,
        mask_format=mask_format,
        image_transform=image_transform,
        audio_transform=audio_transform,
        num_classes=num_classes,
        class_labels=class_labels
    )


def inverse_normalize(tensor):
    inverse_mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
    inverse_std = [1.0/0.229, 1.0/0.224, 1.0/0.225]
    tensor = transforms.Normalize(inverse_mean, inverse_std)(tensor)
    return tensor

def convert_normalize(tensor, new_mean, new_std):
    raw_mean = IMAGENET_DEFAULT_MEAN
    raw_std = IMAGENET_DEFAULT_STD
    # inverse_normalize with raw mean & raw std
    inverse_mean = [-mean/std for mean, std in zip(raw_mean, raw_std)]
    inverse_std = [1.0/std for std in raw_std]
    tensor = transforms.Normalize(inverse_mean, inverse_std)(tensor)
    # normalize with new mean & new std
    tensor = transforms.Normalize(new_mean, new_std)(tensor)
    return tensor