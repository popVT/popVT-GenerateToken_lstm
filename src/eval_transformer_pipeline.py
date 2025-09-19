
from transformers import pipeline
from rouge_score import rouge_scorer
from tqdm import tqdm
import torch

def evaluate_transformer(dataloader, tokenizer, device='cpu', max_examples=50, split_ratio=0.75):
    """
    Оценка distilgpt2 с использованием pipeline
    """
    # Определяем устройство
    device_id = 0 if device == "cuda" else -1
    print(f"Using device: {device_id}")
    
    # Загружаем модель через pipeline
    generator = pipeline(
        "text-generation",
        model="distilgpt2",
        device=device_id,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False  # возвращаем только сгенерированную часть
    )

    # Собираем промпты и целевые тексты
    prompts = []
    target_texts = []
    full_texts = []

    print("Preparing prompts and targets.....")
    for batch in dataloader:
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        for i in range(input_ids.size(0)):
            if len(prompts) >= max_examples:
                break
                
            # Получаем последовательность (игнорируем паддинг)
            sequence = input_ids[i]
            sequence = sequence[sequence != tokenizer.pad_token_id]
                
            # Разделяем на промпт (3/4) и таргет (1/4)
            split_point = int(len(sequence) * split_ratio)
            prompt_tokens = sequence[:split_point]
            target_tokens = sequence[split_point:]                   
                
            # Декодируем
            prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
            target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)
            full_text = tokenizer.decode(sequence, skip_special_tokens=True)
            
            prompts.append(prompt_text)
            target_texts.append(target_text)
            full_texts.append(full_text)
            
        if len(prompts) >= max_examples:
            break

    if not prompts:
        print("Нет подходящих данных для оценки.")
        return None, []

    print(f"Evaluating on {len(prompts)} examples...")
    
    # Инициализация ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    total_scores = {'rouge1': 0.0, 'rouge2': 0.0}
    examples = []

    # Генерация и оценка
    for i, (prompt, target, full) in enumerate(tqdm(zip(prompts, target_texts, full_texts), 
                                                   desc="Evaluating DistilGPT2", 
                                                   total=len(prompts))):
        try:
            # Генерация с различными параметрами
            outputs = generator(
                prompt,
                max_new_tokens=len(target.split()) + 10,  # Длина + запас
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                truncation=True
            )
            
            generated_text = outputs[0]['generated_text'].strip()
            
            # Вычисляем ROUGE между сгенерированной частью и таргетом
            scores = scorer.score(target, generated_text)
            
            # Суммируем scores
            for key in total_scores:
                total_scores[key] += scores[key].fmeasure
            
            # Сохраняем примеры для вывода
            if i < 5:  # Первые 5 примеров
                examples.append({
                    'prompt': prompt[-100:],  # Последние 100 символов промпта
                    'target': target,
                    'generated': generated_text,
                    'rouge1': scores['rouge1'].fmeasure,
                    'rouge2': scores['rouge2'].fmeasure
                })
                
        except Exception as e:
            print(f"Error on example {i}: {e}")
            continue

    # Вычисляем средние значения
    count = len(prompts)
    avg_scores = {key: total_scores[key] / count for key in total_scores}
    
    # Вывод результатов
    print(f"\n{'='*60}")
    print(f"DistilGPT2 Evaluation Results ({count} examples)")
    print(f"{'='*60}")
    print(f"ROUGE-1: {avg_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {avg_scores['rouge2']:.4f}")
    
    # Вывод примеров
    print(f"\n{'='*60}")
    print("Examples:")
    print(f"{'='*60}")
    
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"Prompt: ...{example['prompt']}")
        print(f"Target: {example['target']}")
        print(f"Generated: {example['generated']}")
        print(f"ROUGE-1: {example['rouge1']:.3f}, ROUGE-2: {example['rouge2']:.3f}")
        print("-" * 80)
    
    return avg_scores, examples
