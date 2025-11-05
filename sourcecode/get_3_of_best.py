from datasets import load_from_disk
from transformers import AutoTokenizer
dataset=load_from_disk("./result/c4/0")
selected_score=[item[0] for item in dataset['scores']]
sorted_score=sorted(
    enumerate(selected_score),
    key=lambda x:x[1],
    reverse=True
)
best_score=[score for idx,score in sorted_score[:3]]
selected_idx=[idx for idx,score in sorted_score[:3]]
selected_dataset=dataset.select(selected_idx)
tokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
input_ids=selected_dataset['input_ids']
for idx in range(3):
    text=tokenizer.decode(input_ids[idx],skip_special_tokens=True)
    print(f"[Sample {idx}]{best_score[idx]:.2f}")
    print(text)
