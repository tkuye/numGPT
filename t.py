from gpt import GPT, GPTLayer
from tokenizer import tokenizer
start = "King"
tok = tokenizer()
input_ids = tok.encode(start)
model = GPT.load_model('./checkpoint/model.pkl')

results = model.generate(input_ids, seq_len=25, temperature=0.8, top_k=100)

romeo = tok.decode(results)
print(romeo)


