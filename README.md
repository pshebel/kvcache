python3 -m instruction_following_eval.evaluation_main   --input_data=../kvcache/IFEval/input_data.jsonl --input_response_data=../kvcache/IFEval/control/input_response_data.jsonl --output_dir=../kvcache/IFEval/control/ >> ../kvcache/IFEval/control/results.txt

python3 -m instruction_following_eval.evaluation_main   --input_data=../kvcache/IFEval/input_data.jsonl --input_response_data=../kvcache/IFEval/quantized/input_response_data.jsonl --output_dir=../kvcache/IFEval/quantized/ >> ../kvcache/IFEval/quantized/results.txt

python3 -m instruction_following_eval.evaluation_main   --input_data=../kvcache/IFEval/input_data.jsonl --input_response_data=../kvcache/IFEval/error/input_response_data.jsonl --output_dir=../kvcache/IFEval/error/ >> ../kvcache/IFEval/error/results.txt