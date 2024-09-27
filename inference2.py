# inference2.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデルとトークナイザーのロード
model_name = 'rinna/japanese-gpt2-small'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 1. トークン化と埋め込みの取得
def get_embeddings(input_text):
    # トークン化
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs['input_ids']
    
    # 埋め込み層の重みを取得し、トークンを埋め込みベクトルに変換
    embedding_layer = model.transformer.wte  # Word Token Embedding layer
    embeddings = embedding_layer(input_ids)
    
    return input_ids, embeddings

# 2. 自己注意機構の計算
def self_attention(embeddings, layer_index=0):
    # 指定された層の重みを取得
    attention_layer = model.transformer.h[layer_index].attn
    
    # クエリ、キー、バリューを計算
    query = attention_layer.c_attn(embeddings)[..., :model.config.n_embd]
    key = attention_layer.c_attn(embeddings)[..., model.config.n_embd:model.config.n_embd*2]
    value = attention_layer.c_attn(embeddings)[..., model.config.n_embd*2:]
    
    # アテンションスコアの計算
    scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(model.config.n_embd, dtype=torch.float32))
    attention_weights = torch.nn.functional.softmax(scores, dim=-1)
    
    # 重み付きバリューを計算
    attention_output = torch.matmul(attention_weights, value)
    
    return attention_output

# 3. フィードフォワードネットワークの計算
def feed_forward(attention_output, layer_index=0):
    ff_layer = model.transformer.h[layer_index].mlp
    
    # フィードフォワードの計算
    intermediate_output = torch.nn.functional.relu(ff_layer.c_fc(attention_output))
    ff_output = ff_layer.c_proj(intermediate_output)
    
    return ff_output

# 4. 出力層とソフトマックスの計算
def output_layer(ff_output):
    lm_head = model.lm_head  # 出力層
    logits = lm_head(ff_output)
    
    # ソフトマックスによる次のトークンの確率分布
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    return probs

# 実際にテスト用の文章で各層の計算を実行してみる
if __name__ == '__main__':
    input_text = "これはテストの文章です。"
    
    # 1. トークン化と埋め込み
    input_ids, embeddings = get_embeddings(input_text)
    print(f"Input IDs: {input_ids}")
    print(f"Embeddings: {embeddings}")
    
    # 2. 自己注意機構の計算（最初の層）
    attention_output = self_attention(embeddings, layer_index=0)
    print(f"Attention Output: {attention_output}")
    
    # 3. フィードフォワードネットワークの計算（最初の層）
    ff_output = feed_forward(attention_output, layer_index=0)
    print(f"Feed Forward Output: {ff_output}")
    
    # 4. 出力層とソフトマックスの計算
    probs = output_layer(ff_output)
    print(f"Token Probabilities: {probs}")
