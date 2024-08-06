# 【初心者向け】画像認識 AI を作って犬・猫の判別をしてみた

https://tech.anycloud.co.jp/articles/ai-identification-dog-cat

## ディレクトリ構成

- train
  - dog
  - cat
- validation
  - dog
  - cat
- test.py

※ 学習や検証用の画像は枚数が非常多いため、リポジトリに含めていません。
※ kaggle の[Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats/data?select=train.zip)ページから学習用の画像をダウンロードしてください。

## セットアップ

```
% pip install numpy tensorflow keras pillow
```

## 実行

```
% python test.py
```

## 検証結果の画像

一応、結果の画像を入れておきます。

before: 改善前
after: 改善後

- training-and-validation-accuracy-before.png
- training-and-validation-accuracy-after.png
- training-and-validation-loss-before.png
- training-and-validation-loss-after.png
