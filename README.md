# question_recall_bot

## Contributors
- R12922045 陳沛妤
- R11944078 林資融
- B09611048 蘇蓁葳
- B10705007 劉冠甫

## Environment Setting
- 建置 conda 環境
```
conda create --name adl-final python=3.9
```
- 安裝 pytorch
```
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
- 在根目錄下安裝所需套件
```
pip install -r requirements.txt
```

## Codes Structure and How to run codes
- Question_Recall
	- mt5_three_stage
      - questions_generation
        - inference (change the model to your fine-tuned model & text to your testing context)
        ```
        python qg_inference.py
        ```
        - training_code
        ```
        python run_qg.py qg_config.json
        ```
        - training_data
          - DRCD post processed data(for fine-tuning): DRCD_training_q_gen.json
      - question_answering
        - inference (change the model to your fine-tuned model & question, context to yours)
        ```
        python qa_inference.py
        ```
        - training_code(train_qa.py & utils_qa.py are supporting files for run_qa.py)
        ```
        python run_qa.py qa_config.json
        ```
        - training_data
          - DRCD post processed data(for fine-tuning): DRCD_training_qa.json
      - options_generation
	- mt5_end_to_end
	- LLM_end_to_end
- Dataset
- Streamlit_Demo(change the 3 stage models to your fine-tuned models)
```
streamlit run streamlit_code.py
```
- requirements.txt
- README.md

## Reference
- question_generation & question_answering training code reference to https://github.com/algolet/question_generation/tree/main
