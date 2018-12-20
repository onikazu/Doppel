# Doppel
## 概要
カメラに写った人物が事前に登録された人物と同一人物かを判定するプログラム
slackへの通知機能あり

## 準備
1. [celebA][ca8d3693]をダウンロードし、`big_database_def`配下に置く.その後フォルダ名を`big_database`に変更

  [ca8d3693]: https://www.kaggle.com/jessicali9530/celeba-dataset/home "celebA"

2. 以下のコマンドを入力する(celebAから特徴ベクトルを抽出し、pickle化)
```
$ python image2vector.py
```

3. [slack IncomingWebHook][32e8f276]のURLを`slack_def.ini`に例に習って記述。その後フォルダ名を`slack.ini`に変更

  [32e8f276]: http://slackapi.github.io/node-slack-sdk/reference/IncomingWebhook "slack IncomingWebHook"

4. 登録したい人物の画像を`add_database_def`配下に設置。画像の名前は`(名前)_(id).jpg`という形にする。その後フォルダ名を`add_database`に変更

5. 以下のコマンドを入力する(登録画像の特徴ベクトルを抽出)
```
$ python add_image2vec.py
```

6. 以下のコマンドを入力する
```
$ python doppel_realtime.py
```
