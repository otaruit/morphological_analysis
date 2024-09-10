import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# spaCyモデルのロード
nlp = spacy.load("ja_core_news_sm")  # 日本語モデル

# テキスト読み込み
file_path = 'rasyo-mon.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# データをPandasのDataFrameに格納
df = pd.DataFrame(lines, columns=['テキスト'])

# 不要な空行や空白行の削除
df['テキスト'] = df['テキスト'].str.strip()  # 空白を削除
df = df[df['テキスト'] != '']  # 空白行の削除

# 名詞のみを抽出する関数
def extract_nouns(text):
    doc = nlp(text)
    return ' '.join([token.text for token in doc if token.pos_ == 'NOUN'])

# DataFrameに名詞のみを格納する列を作成
df['名詞のみ'] = df['テキスト'].apply(extract_nouns)

# TF-IDFを用いてベクトル化し、キーワードを抽出
vectorizer = TfidfVectorizer(stop_words=['こと', 'ため', 'よう', 'もの', 'これ', 'それ', 'どこ', 'そこ', 'たい', 'ほか', 'さっき', 'びと'])
X = vectorizer.fit_transform(df['名詞のみ'])

# ベクトル化されたデータをDataFrameに変換
df_vectorized = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# TOP30のキーワードを特定
top_30_keywords = df_vectorized.sum().nlargest(30)
print("TOP30のキーワード:")
print(top_30_keywords)
