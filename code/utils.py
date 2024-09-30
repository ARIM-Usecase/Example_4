# 汎用ライブラリ
from tqdm import tqdm
from PIL import Image
from glob import glob

from statistics import mean

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from datasets import SEMDataset

# 正規化のための平均と標準偏差を計算する関数
def calculate_statistics(folder):
    """
    指定されたフォルダー内の画像の平均と標準偏差を計算する関数。

    Args:
        folder (list): 画像ファイルのパスのリスト。

    Returns:
        tuple: 画像の平均値と標準偏差のリスト（それぞれ3要素のリスト）。
    """
    avg, std = list(), list()
    for img in tqdm(folder):
        img = Image.open(img)
        tensor = TF.to_tensor(img)
        tensor = tensor.float()
        avg.append(torch.mean(tensor).tolist())
        std.append(torch.std(tensor).tolist())
    avg = [round(mean(avg), 3)] * 3
    std = [round(mean(std), 3)] * 3

    return avg, std

# データをDatasetクラスに読み込む関数
def load_datasets(config):
    """
    設定に基づいてトレーニングおよび検証データセットを読み込む関数。

    Args:
        config (dict): データセットの設定情報を含む辞書。

    Returns:
        tuple: トレーニングセットと検証セット。
    """
    data_dir = config['data_dir']
    if config['normalize']:
        train_avg, train_std = calculate_statistics(glob(f'./{data_dir}/train/*/*.jpg'))
        train_img_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=train_avg, std=train_std)
        ])
        
        validate_avg, validate_std = calculate_statistics(glob(f'./{data_dir}/validate/*/*.jpg'))
        validate_img_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=validate_avg, std=validate_std)
        ])
        
        train_set = SEMDataset(data_dir+'/train', train_img_transforms, seed=config['seed'])
        val_set = SEMDataset(data_dir+'/validate', validate_img_transforms, seed=config['seed'])
        
    else:
        train_img_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
        train_set = SEMDataset(data_dir+'/train', train_img_transforms, seed=config['seed'])
        validate_img_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
        val_set = SEMDataset(data_dir+'/validate', validate_img_transforms, seed=config['seed'])
        
    return train_set, val_set

def load_test_dataset(config):
    """
    設定に基づいてテストデータセットを読み込む関数。

    Args:
        config (dict): データセットの設定情報を含む辞書。

    Returns:
        SEMDataset: テストデータセット。
    """
    data_dir = config['data_dir']
    if config['normalize']:
        test_avg, test_std = calculate_statistics(glob(f'./{data_dir}/test/*/*.jpg'))
        test_img_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=test_avg, std=test_std)
        ])
        
        test_set = SEMDataset(data_dir+'/test', test_img_transforms, seed=config['seed'])
        
    else:
        test_img_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
        test_set = SEMDataset(data_dir+'/test', test_img_transforms, seed=config['seed'])
 
    return test_set

def loss_fn(config):
    """
    設定に基づいて損失関数を選択する関数。

    Args:
        config (dict): 設定情報を含む辞書。

    Returns:
        torch.nn.Module: 選択された損失関数のインスタンス。

    Raises:
        NotImplementedError: サポートされていない損失関数が指定された場合。
    """

    if config['criterion'] == 'CrossEntropyLoss':
        return torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

def optimize_fn(config, parameters):
    """
    設定に基づいてオプティマイザーを選択する関数。

    Args:
        config (dict): 設定情報を含む辞書。
        parameters (iterable): 最適化するパラメータのリスト。

    Returns:
        torch.optim.Optimizer: 選択されたオプティマイザーのインスタンス。

    Raises:
        NotImplementedError: サポートされていないオプティマイザーが指定された場合。
    """

    if config['optimizer'] == 'Adam':
        return torch.optim.Adam(parameters, config['lr'])
    else:
        raise NotImplementedError

def binary_acc(y_pred, y_test):
    """
    二項分類の精度を計算する関数。

    Args:
        y_pred (torch.Tensor): モデルの予測結果。
        y_test (torch.Tensor): 正解ラベル。

    Returns:
        torch.Tensor: 計算された精度（パーセント）。
    """

    y_pred_tag = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_tag, dim=1)
    correct_results_sum = (y_pred_tags == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc

def class_acc(y_pred, y_test):
    """
    クラス分類の精度を計算する関数。

    Args:
        y_pred (torch.Tensor): モデルの予測結果。
        y_test (torch.Tensor): 正解ラベル。

    Returns:
        torch.Tensor: 計算された精度（パーセント）。
    """

    y_pred_tag = torch.softmax(y_pred, dim=1)
    y_pred_tags = torch.argmax(y_pred_tag, dim=1)
    correct_results_sum = (y_pred_tags == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc
