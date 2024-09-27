# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from inference2 import load_model, generate_text

app = FastAPI()

# Load the model
local_model_path = 'rinna/japanese-gpt2-small'
generator = load_model(local_model_path)

class InputText(BaseModel):
    input_text: str

@app.post("/generate")
def generate(input_data: InputText):
    # Generate text
    output = generate_text(generator, input_data.input_text)
    return output


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=80)