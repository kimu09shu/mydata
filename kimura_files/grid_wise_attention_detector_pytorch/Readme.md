# Grid-wise-attention
製作者 木村秋斗

## 環境構築　

Dockerfileで環境を構築

- python 3.7
- pytorch 1.8.0
- cuda 11.1.1
- cudnn 8.0


##使用法
環境構築後，以下のコマンドを実行
```bash
cd utils
make all 
```
###データセットのディレクトリ構造
学習，評価にはCityPersonsデータセットを使用
./data/citypersons に以下の構造でデータを格納
```bash
--- citypersons
    |    
    |--- images
    |    |--- train
    |    |---val
    |
    |--- annotations
         |--- anno_train.mat
         |--- anno_val.mat
 
```
###学習
学習は以下のコマンドを実行
```bash
python train.py 
```
parser
- --detection_model：検出方法（anchor：アンカーによる検出，retina：retinanetの方法で検出，csp：Center and Scale Prediction 対象の中心とスケールによる検出）
- --detection_scale： スケール（single：単一のスケール，multi：マルチスケール）
- --ngpu：gpuの数
- --resume_epoch：学習を再開する際の開始epoch数

###評価
評価は以下のコマンドを実行
```bash
python val.py　
```
parser
- --detection_model：検出方法（anchor：アンカーによる検出，retina：retinanetの方法で検出，csp：Center and Scale Prediction 対象の中心とスケールによる検出）
- --detection_scale： スケール（single：単一のスケール，multi：マルチスケール）
- --model_dir：学習済みモデルのパス
- --model_path：学習済みモデルのパス

val.pyのvalクラスはtrainval.pyのvalクラスのみの実行と同様に，指定した学習済みモデルを使用して評価．

val_epochクラスは各epochごとに評価を行う
####attention mapの可視化
評価時に可視化を行っても良いが，総画素数分のattention mapが生成されることにより処理時間が膨大になるため，画像1枚ごとの実行を推奨
```bash
python test.py
```
parser
- --detection_model：検出方法（anchor：アンカーによる検出，retina：retinanetの方法で検出，csp：Center and Scale Prediction 対象の中心とスケールによる検出）
- --detection_scale： スケール（single：単一のスケール，multi：マルチスケール）
- --model_dir：学習済みモデルのパス
- --model_path：学習済みモデルのパス
- --image_path：入力する画像のパス

###configについて

主なconfigは実行するコード(train.py，val.py，test.py)で設定できるようにしてあります
- train_path：学習データセットのパス
- gpu_ids： 使用するGPU番号
- onegpu：1つのGPUの処理数（バッチサイズ = gpu_ids * onegpu）
- size_train：学習時の画像サイズ
- init_lr：学習率
- num_classes：クラス数
- rois_thresh：推論時の閾値
