import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import numpy as np

# モデルとトークナイザーをグローバルにロードしてキャッシュする
model_name = 'rinna/japanese-gpt2-small'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 1. トークン化と埋め込みの取得
def get_embeddings(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs['input_ids']
    embedding_layer = model.transformer.wte
    embeddings = embedding_layer(input_ids)
    return input_ids, embeddings

# 2. 自己注意機構の計算
def self_attention(embeddings, layer_index=0):
    attention_layer = model.transformer.h[layer_index].attn
    hidden_size = model.config.n_embd
    c_attn = attention_layer.c_attn
    mixed_x_layer = c_attn(embeddings)
    query, key, value = mixed_x_layer.split(hidden_size, dim=2)
    
    scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(hidden_size, dtype=torch.float32))
    attention_weights = torch.nn.functional.softmax(scores, dim=-1)
    
    attention_output = torch.matmul(attention_weights, value)
    return attention_output, attention_weights

# 3. フィードフォワードネットワークの計算
def feed_forward(attention_output, layer_index=0):
    ff_layer = model.transformer.h[layer_index].mlp
    intermediate_output = torch.nn.functional.relu(ff_layer.c_fc(attention_output))
    ff_output = ff_layer.c_proj(intermediate_output)
    return ff_output

# 4. 出力層とソフトマックスの計算
def output_layer(ff_output):
    lm_head = model.lm_head
    logits = lm_head(ff_output)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

# アテンションウェイトのヒートマップ表示
def visualize_attention(attention_weights, input_tokens):
    attn = attention_weights[0, 0].detach().numpy()
    tokens = [tokenizer.decode([token_id]) for token_id in input_tokens[0]]
    plt.figure(figsize=(8, 6))
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
    plt.xlabel('キー（入力トークン）')
    plt.ylabel('クエリ（出力トークン）')
    plt.title('アテンションウェイトのヒートマップ')
    plt.savefig("/workspace/output/attention_weights.png") 
    plt.show()

# 埋め込みベクトルの2Dプロット
def visualize_embeddings(embeddings, input_tokens):
    embeddings_np = embeddings[0].detach().numpy()
    tokens = [tokenizer.decode([token_id]) for token_id in input_tokens[0]]
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_np)
    
    plt.figure(figsize=(8, 6))
    for i, token in enumerate(tokens):
        x, y = embeddings_2d[i]
        plt.scatter(x, y)
        plt.text(x+0.01, y+0.01, token, fontsize=12)
    plt.title('埋め込みベクトルの2Dプロット')
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    plt.savefig("/workspace/output/embeddings.png")   # 画像をファイルに保存
    plt.show()

# 正誤応答の比較
def compare_responses(input_text_correct, input_text_incorrect):
    # 正しい応答
    input_ids_correct, embeddings_correct = get_embeddings(input_text_correct)
    attention_output_correct, attention_weights_correct = self_attention(embeddings_correct)
    
    # 誤った応答
    input_ids_incorrect, embeddings_incorrect = get_embeddings(input_text_incorrect)
    attention_output_incorrect, attention_weights_incorrect = self_attention(embeddings_incorrect)
    
    # 埋め込みベクトルの視覚化
    visualize_embeddings(embeddings_correct, input_ids_correct)
    
    # アテンションウェイトの視覚化
    visualize_attention(attention_weights_correct, input_ids_correct)

# 実行
if __name__ == '__main__':
    input_text_correct = "これはテストの文章です。"
    input_text_incorrect = "これは誤ったテキストです。"
    
    # 正誤応答の比較
    compare_responses(input_text_correct, input_text_incorrect)
