import os
import random
from glob import glob

from PIL import Image

import torch
from torch.utils.data import Dataset

class SEMDataset(Dataset):

    """
    SEMDatasetは、指定されたディレクトリ内の画像データセットを管理するためのクラスです。
    
    Attributes:
        img_root (str): 画像データが格納されているルートディレクトリのパス。
        mode (str): ディレクトリ名。
        labels (list): ディレクトリ内のラベル（サブディレクトリ名）のリスト。
        img_names (list): 読み込む画像ファイルのパスのリスト。
        transform_img (callable): 画像に適用する変換関数。
        seed (int): ランダムシードの値。
        class2idx (dict): ラベルからインデックスへのマッピング辞書。
        idx2class (dict): インデックスからラベルへのマッピング辞書。
    """
        
    def __init__(self, root, transform_img, seed = 3543032):
        
        """
        SEMDatasetの初期化メソッド。
        
        Args:
            root (str): 画像データのルートディレクトリのパス。
            transform_img (callable): 画像に適用する変換関数。
            seed (int, optional): ランダムシードの値（デフォルトは3543032）。
        """
        self.img_root = os.path.join('./', root)
        self.mode = os.path.basename(self.img_root)
        self.labels = sorted([f for f in os.listdir(
            self.img_root) if os.path.isdir(os.path.join(self.img_root, f))])
        self.img_names = glob(os.path.join(self.img_root, '*/*.jpg'))

        self.transform_img = transform_img
        self.seed = seed
        self.class2idx = {}
        self.idx2class = {}
        
        for i in range(len(self.labels)):
            self.class2idx[self.labels[i]] = i
            self.idx2class[i] = self.labels[i]

    def __getitem__(self, index):

        """
        指定されたインデックスに基づいて画像とラベルを取得するメソッド。
        
        Args:
            index (int): 取得するアイテムのインデックス。
        
        Returns:
            tuple: 変換された画像、対応するラベルのテンソル、画像ファイル名のタプル。
        """
                
        img_name = self.img_names[index]

        # Seed the random generator
        random.seed(self.seed)
        img = self.transform_img(Image.open(img_name).convert('RGB'))

        label = os.path.basename(os.path.dirname(img_name))
        return img, torch.tensor(self.class2idx[label]), os.path.basename(img_name)

    def __len__(self):

        """
        データセットのサイズを返すメソッド。
        
        Returns:
            int: データセット内の画像の数。
        """
                
        return len(self.img_names)
