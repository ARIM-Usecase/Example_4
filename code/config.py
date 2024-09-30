"""
ネットワークのトレーニング設定を決定します。このモジュールは、トレーニングプロセスで
各データセットに定義されたバージョンからハイパーパラメータを取得するために使用されます。
このバージョンは、ネットワークのトレーニング、評価、および適用時に指定する必要があります。
ユーザーが新しいデータセットでトレーニングを行うか、新しいハイパーパラメータを使用したい場合、
このモジュールを更新する必要があります。
（これはテキストエディタまたはコードエディタで行うのが最も簡単であり、コマンドラインからは
行わない方が良いです。）
"""

import sys


# 指定されたデータセットのデフォルト設定を生成します。
# 次のハイパーパラメータが含まれています。これらのデフォルト値は、下記のget_config関数で
# 指定されたバージョンによって後で更新されます。

def gen_default(dataset, size, batch_size=4, lr=1e-4, epoch=50):
    """
    データセットのデフォルト設定を生成する関数。

    Args:
        dataset (str): データセットの名前。
        size (int): データセットのサイズ。
        batch_size (int, optional): バッチサイズ（デフォルトは4）。
        lr (float, optional): 学習率（デフォルトは1e-4）。
        epoch (int, optional): エポック数（デフォルトは10）。

    Returns:
        dict: デフォルト設定の辞書。
    """
    default = {
        'data_dir': './data/' + dataset,
        'size': size,
        'batch_size': batch_size,
        'optimizer': 'Adam',
        'lr': lr,
        'epoch': epoch,
        'normalize': False,
        'pretrained': True,
        'criterion': 'CrossEntropyLoss',
        'seed': 42,
    }
    
    return default

config = {
    'example': {
        'default': gen_default('example', size=10),
        'v1': {'lr': 0.00035446559318532957}
    }
}

# get_config関数は、gen_default関数によって提供されたデフォルトのハイパーパラメータの
# セットを取得し、指定されたバージョンに応じて更新または新しいエントリを作成します。

def get_config(dataset, version):
    """
    指定されたデータセットとバージョンに基づいて設定を取得する関数。

    Args:
        dataset (str): データセットの名前。
        version (str): 使用する設定のバージョン。

    Returns:
        dict: 更新されたハイパーパラメータの辞書。

    Raises:
        SystemExit: 指定されたデータセットが存在しない場合。
    """
    try:
        args = config[dataset]['default'].copy()
    except KeyError:
        print(f'データセット {dataset} は存在しません')
        sys.exit(1)
    try:
        args.update(config[dataset][version])
    except KeyError:
        print(f'バージョン {version} は定義されていません')

    args['name'] = dataset + '_' + version

    return args
