from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "./results/checkpoint-125"   # 👈 path to your trained checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Make sure model is on the right device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Example sentences
examples = [
    "This is a small test.",
    "I am learning how to train translation models.",
    "Good morning, how are you?"
]

def inference_translate(tokenizer = tokenizer, model = model, device = device, examples = examples):


    for text in examples:
    # Prepend the prefix T5 expects
        input_text = "translate English to Spanish: " + text
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate translation
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=120, num_beams=4)

    # Decode and print
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"EN: {text}")
        print(f"ES: {translation}\n")


