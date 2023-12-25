import json
import random

# read in the data
with open('./DRCD_test.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

print(data["data"][0])

output = []

# "paragraphs": [
#     {
#         "context": "2010年引進的廣州快速公交運輸系統，屬世界第二大快速公交系統，日常載客量可達100萬人次，高峰時期每小時單向客流高達26900人次，僅次於波哥大的快速交通系統，平均每10秒鐘就有一輛巴士，每輛巴士單向行駛350小時。包括橋樑在內的站台是世界最長的州快速公交運輸系統站台，長達260米。目前廣州市區的計程車和公共汽車主要使用液化石油氣作燃料，部分公共汽車更使用油電、氣電混合動力技術。2012年底開始投放液化天然氣燃料的公共汽車，2014年6月開始投放液化天然氣插電式混合動力公共汽車，以取代液化石油氣公共汽車。2007年1月16日，廣州市政府全面禁止在市區內駕駛摩托車。違反禁令的機動車將會予以沒收。廣州市交通局聲稱禁令的施行，使得交通擁擠問題和車禍大幅減少。廣州白雲國際機場位於白雲區與花都區交界，2004年8月5日正式投入運營，屬中國交通情況第二繁忙的機場。該機場取代了原先位於市中心的無法滿足日益增長航空需求的舊機場。目前機場有三條飛機跑道，成為國內第三個擁有三跑道的民航機場。比鄰近的香港國際機場第三跑道預計的2023年落成早8年。",
#         "id": "1001-10",
#         "qas": [
#             {
#                 "id": "1001-10-1",
#                 "question": "廣州的快速公交運輸系統每多久就會有一輛巴士？",
#                 "answers": [
#                     {
#                         "id": "1",
#                         "text": "10秒鐘",
#                         "answer_start": 84
#                     }
#                 ]
#             },
#             {
#                 "id": "1001-10-2",
#                 "question": "從哪一天開始在廣州市內騎摩托車會被沒收？",
#                 "answers": [
#                     {
#                         "id": "1",
#                         "text": "2007年1月16日",
#                         "answer_start": 256
#                     }
#                 ]
#             },
#             {
#                 "id": "1001-10-3",
#                 "question": "廣州白雲國際機場在完成第三條跑道的後八年哪一座機場也會有第三跑道？",
#                 "answers": [
#                     {
#                         "id": "1",
#                         "text": "香港國際機場",
#                         "answer_start": 447
#                     }
#                 ]
#             }
#         ]
#     },

# [{'id': 'c398789b7375e0ce7eac86f2b18c3808',
#    'question': '隐藏式行车记录仪哪个牌子好',
#    'context': '推荐使用360行车记录仪。行车记录仪的好坏，取决于行车记录仪的摄像头配置，配置越高越好，再就是性价比。 行车记录仪配置需要1296p超高清摄像头比较好，这样录制视频清晰度高。再就是价格，性价比高也是可以值得考虑的。 360行车记录仪我使用了一段时间 ，觉得360行车记录仪比较好录得广角比较大，并且便宜实惠 ，价格才299，在360商城可以买到。可以参考对比下。',
#    'answers': {'answer_start': [4], 'text': ['360行车记录仪']}}]

num = 0

for d in data["data"]:
    for p in d["paragraphs"]:
        output_dict = {}
        output_dict["context"] = p["context"]
        # random select one question
        question_len = len(p["qas"])
        random_index = random.randint(0, question_len-1)
        output_dict["question"] = p["qas"][random_index]["question"]
        output_dict["id"] = p["qas"][random_index]["id"]
        answer = p["qas"][random_index]["answers"][0]
        format_answer = {}
        format_answer["answer_start"] = [answer["answer_start"]]
        format_answer["text"] = [answer["text"]]
        output_dict["answers"] = format_answer
        output.append(output_dict)
        num += 1

print(num)
# write out the data
with open('./DRCD_test_qa.json', 'w', encoding='utf-8') as json_file:
    json.dump(output, json_file, ensure_ascii=False)