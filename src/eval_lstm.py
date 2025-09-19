import torch
import evaluate
from rouge_score import rouge_scorer

def model_eval(model, val_dataloader, tokenizer, device):
    #rouge = evaluate.load('rouge')
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)

    model.eval()
    total_rouge1 = 0
    total_rouge2 = 0
    count = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            for i in range(input_ids.size(0)):
                seg_len = int(input_ids.size(0) * 0.75)

                promt_ids = input_ids[i][: seg_len]
                promt = tokenizer.decode(promt_ids, skip_special_tokens=True)

                target_ids = labels[i][seg_len:]
                target_ids = target_ids[target_ids != tokenizer.pad_token_id]  # Убираем паддинг
                target = tokenizer.decode(target_ids, skip_special_tokens=True)

                generated = model.generate(tokenizer, promt, max_length=30, device=device)
                
                results = scorer.score(target, generated)

                #results = rouge.compute(predictions=target, references=generated, use_stemmer=True)
                total_rouge1 += results['rouge1'].fmeasure
                total_rouge2 += results['rouge2'].fmeasure
                count += 1

    rouge1 = total_rouge1 / count
    rouge2 = total_rouge2 / count

    print(f"LSTM ROUGE-1: {rouge1:.4f}")
    print(f"LSTM ROUGE-2: {rouge2:.4f}")

    return rouge1, rouge2



