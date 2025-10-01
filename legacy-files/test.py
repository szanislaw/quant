from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "microsoft/phi-3-mini-4k-instruct"
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tok)

out = pipe("Explain what RSI > 70 means in trading:", max_new_tokens=50)
print(out[0]["generated_text"])

