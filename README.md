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
	  	- inference (change the model to your fine-tuned model & text to your testing context)
	        ```
	        python src/inference.py --model_name_or_path "./model"  --test_file path/to/test.json --text_column instruction \
   	     	--num_beams 5   --output_dir ./ --overwrite_output_dir true --predict_with_generate true
	        ```
	        - training_code
	        ```
	        python train.py --model_name_or_path="google/mt5-base"  --num_beams=5
   	     	--train_file="path/to/train_m2_mt5.json" --text_column="instruction" --summary_column="output" \
   	     	--preprocessing_num_workers=6 --output_dir="./model" --do_train  --num_train_epochs=10 --auto_find_batch_size \
   	     	--learning_rate=4e-5 --gradient_accumulation_steps=4 --overwrite_output_dir
	        ```
	        - training_data
	          - C3 processed data(for fine-tuning): train_m2_mt5.json
	- LLM_end_to_end
   		- inference (change the model to your fine-tuned model & text to your testing context)
	        ```
         	python guanaco_generate.py --model_name_or_path /path/to/Taiwan-LLaMa-folder --adapter_path /path/to/adapter_checkpoint \
         	--input_file_path /path/to/input.json  --output_file_path /path/to/output.json
	        ```
	        - training_code
	        ```
	        python qlora.py --model_name_or_path /path/to/Taiwan-LLaMa-folder --dataset_format input-output --dataset /path/to/train_m.json \
         	--max_train_samples 7000 --max_steps 500  --do_eval True --save_steps 100
	        ```
	        - training_data
	          - C3 processed data(for fine-tuning): train_m.json
- Dataset
- Streamlit_Demo(change the 3 stage models to your fine-tuned models)
```
streamlit run streamlit_code.py
```
- requirements.txt
- README.md

## Reference
- question_generation & question_answering training code reference to https://github.com/algolet/question_generation/tree/main
- options_generation training code reference to https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py
