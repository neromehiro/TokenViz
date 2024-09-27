# Dockerfile
FROM sonoisa/deep-learning-coding:pytorch1.12.0_tensorflow2.9.1

# Set the working directory
WORKDIR /workspace

# Install required packages
COPY requirements.txt /workspace/
RUN pip install -r requirements.txt

# Pre-download the model to /workspace/models
RUN mkdir -p /workspace/models/rinna
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('rinna/japanese-gpt2-small', cache_dir='/workspace/models/rinna'); \
    AutoModelForCausalLM.from_pretrained('rinna/japanese-gpt2-small', cache_dir='/workspace/models/rinna')"

# Copy application code
COPY . /workspace/

# Expose the port Flask will run on
EXPOSE 8080

# Run the Flask app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
