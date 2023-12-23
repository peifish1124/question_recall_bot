from question_generation import pipeline

question_answerer = pipeline("question-answering", model="../training_result/qa", device="cuda")

question = '在寒冷的冬天,農夫在哪裡發現了一條可憐的蛇?'
context = "在一個寒冷的冬天，趕集完回家的農夫在路邊發現了一條凍僵了的蛇。他很可憐蛇，就把它放在懷裡。當他身上的熱氣把蛇溫暖以後，蛇很快甦醒了，露出了殘忍的本性，給了農夫致命的傷害──咬了農夫一口。農夫臨死前說：「我竟然救了一條可憐的毒蛇，就該受到這種報應啊！"

qa_result = question_answerer(input={'question': question, 'context': context})
print(qa_result)