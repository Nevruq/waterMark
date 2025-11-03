import Kirchenheimer

model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Beispiel (nimm ein verfügbares)
tok = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

secret = b"my_super_secret_key_32bytes_min"   # sicher speichern
gamma  = 0.5
delta  = 2.5

proc = WatermarkLogitsProcessor(
    tokenizer=tok,
    secret_key=secret,
    gamma=gamma,
    delta=delta,
    green_exclude_ids=tok.all_special_ids
)
processors = LogitsProcessorList([proc])

prompt = "Explain the basics of symmetric and asymmetric encryption."
inputs = tok(prompt, return_tensors="pt").to(model.device)

out = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=1.0,
    top_p=0.9,
    do_sample=True,
    logits_processor=processors
)

text = tok.decode(out[0], skip_special_tokens=True)
print(text)
