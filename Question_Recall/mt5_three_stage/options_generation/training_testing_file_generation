import opencc
import json
from tqdm import tqdm
import jsonlines
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='c3-m-test.json')
    parser.add_argument('--output_file', type=str, default='choice_gen_v5_test.jsonl')
    args = parser.parse_args()
    return args
def main(input_file, output_file):

    converter = opencc.OpenCC('s2twp.json') # convert from Simplified Chinese to Traditional Chinese
    with open(input_file, 'r', encoding = 'utf-8') as f:
        data = json.load(f)

    formatted = []

    idi = 0
    for i in tqdm(data):
        for j in i[1]:
            try:
                choice = j['choice'].remove(j['answer'])
                inputStr = f'''{j['question']} <extra_id_0> {j['answer']}'''
                outputStr = f'''{j['answer']} <extra_id_1> {j['choice'][0]} <extra_id_2> {j['choice'][1]} <extra_id_3> {j['choice'][2]}'''
                formatted.append(
                {
                    'id': idi,
                    'split': 'train',
                    'qa': converter.convert(inputStr),
                    'choices': converter.convert(outputStr),
                }
                )
                idi += 1
            except:
                pass

    file_path = output_file

    # Dumping data into a JSON Lines file
    with jsonlines.open(file_path, mode='w') as writer:
        writer.write_all(formatted)

if __name__ == '__main__':
    args = parse_args()
    main(args.input_file, args.output_file)