# Center and Scale Prediction + attention

## 環境構築　

Dockerfileで環境を構築

- python 3.7
- pytorch 1.2.0
- cuda 9.0
- cudnn 7.0


##使用法
環境構築後，以下のコマンドを実行
```bash
cd util
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
python trainval.py --ABN_model 'LAP' --LAP_type 'sliding_window' --validation True
```

- --ABN_model：使用するattention構造（LAP：Local attention pooling，GAM：Attention-Mask機構，None：attention構造未使用）
- --LAP_type： LAPで使用する構造（sliding_window：各対象画素の周囲を範囲としたattentionの取得，fixed_area：画像全体を分割した範囲の各画像に対するattentionの取得）
- --validation：epochごとに評価を行うかどうか

###評価
評価はtrainval.pyのvalクラスのみを実行するか，以下のコマンドを実行
```bash
python val.py　--ABN_model 'LAP' --LAP_type 'sliding_window' --model_dir './ckpt-LAP' --model_path 'CSPNet-150.pth'
```
- --model_dir：学習済みモデルのパス
- --model_path：学習済みモデルのパス

val.pyのvalクラスはtrainval.pyのvalクラスのみの実行と同様に，指定した学習済みモデルを使用して評価．

val_epochクラスは各epochごとに評価を行う
####attention mapの可視化
評価時に可視化を行っても良いが，LAPでは総画素数分のattention mapが生成されることにより処理時間が膨大になるため，画像1枚ごとの実行を推奨
```bash
python test.py　--ABN_model 'LAP' --LAP_type 'sliding_window' --image_path './data/citypersons/images/val/frankfurt_000001_005898_leftImg8bit.png'  --model_dir './ckpt-LAP' --model_path 'CSPNet-150.pth'
```
- --image_path：入力する画像のパス

