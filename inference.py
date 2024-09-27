# inference.py
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_model(model_path):
    # ワークスペースにキャッシュされたモデルをロード
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir='/workspace/models/rinna')
    model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir='/workspace/models/rinna')
    return pipeline('text-generation', model=model, tokenizer=tokenizer)

def generate_text(generator, input_text, max_length=50, num_return_sequences=1):
    # テキスト生成
    output = generator(input_text, max_length=max_length, num_return_sequences=num_return_sequences)
    return output

if __name__ == '__main__':
    model_name = 'rinna/japanese-gpt2-small'
    generator = load_model(model_name)
    
    # テスト用の入力
    input_text = "これはテストの文章です。"
    output = generate_text(generator, input_text)
    
    print(output)
