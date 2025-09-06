from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# load model once
def load_llm():
    model_name = "microsoft/phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype="auto"
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

llm = load_llm()

def llm_commentary(ticker: str, context: str) -> str:
    system_prompt = f"""
    You are an expert options analyst.
    Analyze the latest signals for {ticker} and provide exactly:
    1. Upside Potential
    2. Risks / Downside
    3. Suggested Action
    """

    prompt = f"{system_prompt}\n\nContext:\n{context}"
    result = llm(prompt, max_new_tokens=400, temperature=0.7)
    return result[0]["generated_text"].strip()
