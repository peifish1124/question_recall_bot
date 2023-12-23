import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("../training_result/qg_small")
model = AutoModelForSeq2SeqLM.from_pretrained("../training_result/qg_small")
model.eval()

text = "今天下午小英在圖書館看了一部電影，這部電影叫做「旅館主人」，講的是一個人怎麼成功的故事。\
        電影中的人本來是一個只有小學畢業、沒有工作、身上只剩一百塊錢的人，後來卻變成一家旅館的老闆。\
        旅館的生意非常好，所以他變得很有錢。這原來是一本小說的故事，因為這本小說寫得很棒，就被拍成電影了。"

text = "question generation: " + text
inputs = tokenizer(text,
                   return_tensors='pt',
                   truncation=True,
                   max_length=512)

with torch.no_grad():
  outs = model.generate(input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=128,
                        no_repeat_ngram_size=4,
                        num_beams=4)

question = tokenizer.decode(outs[0], skip_special_tokens=True) 
questions = [q.strip() for q in  question.split("<sep>") if len(q.strip()) > 0]
print(questions)
