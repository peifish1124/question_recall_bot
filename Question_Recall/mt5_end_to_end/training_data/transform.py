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
        context =   context 
        for _,conversation in enumerate(content[1][1]):
            time += 1
            question_data = conversation['question'] 
            answer = conversation['answer']
            output_str = ""
            k = 2
            for j, choice in enumerate(conversation['choice']):
                if choice == answer:
                    continue
                k += 1
                output_str += f'<extra_id_{k}>:{choice}, '
            output_str = output_str[:-2]
            output_str = f'<extra_id_1>{question_data} <extra_id_2> {answer} {output_str} \n'
            output = {
                "id": time,
                "instruction": context,
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

