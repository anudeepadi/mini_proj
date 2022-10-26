
"""
This script downloads the model from the huggingface model hub.
"""
from transformers import pipeline

pipe = pipeline(
    task = "summarization",
    model = "facebook/bart-large-cnn"
)

pipe.save_pretrained("model")

print("Downloaded All Models!")