🐶🐱 Dog-Cat Classifier  
ブラウザから画像をアップロード → 即座に犬か猫かを判定するAI Webアプリ  
PyTorchで学習させたモデルをFlaskでWeb化し、レスポンシブで見やすいUIを実装しました。

![アプリのデモ](images/demo.gif)
アップロード → 判定結果表示までのデモ

✨ 特徴  
・高精度分類：テスト精度95%以上  
・超高速推論：画像アップロードから結果表示まで1秒以内  
・見やすい結果表示：犬は青、猫は赤で判定結果を色分け  
・レスポンシブ対応：PC・スマホ両方で快適動作  
・シンプルな操作：画像を選んでボタンを押すだけ  

🛠 使用技術
| カテゴリ | 技術 |
| :--- | :--- |
| AI モデル | PyTorch, Torchvision |
| バックエンド | Flask |
| フロントエンド | HTML, CSS, JavaScript |
| 環境構築 | Python venv |
| デザイン | レスポンシブCSS, カラーコーディング |

🚀 実行方法
1. 仮想環境のセットアップ
python -m venv venv312
source venv312/bin/activate  # macOS/Linux
venv312\Scripts\activate     # Windows
2. 必要なライブラリをインストール
pip install -r requirements.txt
3. モデルの学習（任意）
python train_model.py
4. Web アプリケーションを起動
python app.py
ブラウザで http://127.0.0.1:5000 にアクセス

📊 モデル性能  
・テスト精度：95.2%  
・テスト損失：0.0728  
・小規模データセットながら安定した高精度を実現  

📌 今後の展望  
・マルチクラス分類への拡張（犬・猫以外の動物も判定）  
・クラウドデプロイ（Heroku / Render / AWS）  
・モバイルアプリ化（React Native or Flutter）  

📄 ライセンス  
このプロジェクトは[MIT License](LICENSE)の下で公開されています。  