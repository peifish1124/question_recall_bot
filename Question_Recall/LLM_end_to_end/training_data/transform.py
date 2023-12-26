import json
from opencc import OpenCC

def convert_data(input_data):
    converted_data = []
    time = 0
    for  content in enumerate(input_data):

        tmp = content[1][0]
        context = []
        for _ , c in enumerate(tmp):
            context.append(c+"\n")
        context = "".join(context)
        context =  "你是能根據文章來生出問題、選項以及對應答案的專家，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、明確、詳細的回答。\
USER: 請針對以下文章生出一個問題、四個選項以及一個正確答案，且需要確保四個選項中只有一個是正確答案，而其他三個是錯誤答案。 文章:"+ context +" ASSISTANT:"
        for _,conversation in enumerate(content[1][1]):
            time += 1
            question_data = conversation['question'] 
            answer = conversation['answer']
            output_str = ""
            for j, choice in enumerate(conversation['choice']):
                output_str += f"{chr(65 + j)}:{choice}, "
            output_str = output_str[:-2]
            output_str = f"Question: {question_data}, \n Choice: {output_str} \n Answer: {answer}"
            output = {
                "id": time,
                "instruction": context,
                "output":  output_str,
            }
            converted_data.append(output)
    return converted_data

# Assuming you have already loaded the data from the file
with open("path/to/file.json", "r") as f:
     input_data = eval(f.read())

converted_data = convert_data(input_data)

# Convert to Traditional Chinese using OpenCC
converter = OpenCC('s2tw')  # Simplified Chinese to Traditional Chinese
converted_data_chinese = json.dumps(converted_data, indent=4, ensure_ascii=False)
converted_data_chinese = converter.convert(converted_data_chinese)

with open("path/to/file.json", "w") as f:
    f.write(converted_data_chinese)

