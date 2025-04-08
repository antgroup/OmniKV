config=baselines/infllm/config/llama-3-inf-llm.yaml

conda deactivate
source /envs/fast/bin/activate

datasets="narrativeqa,qasper,multifieldqa_en,\
hotpotqa,2wikimqa,musique,\
gov_report,qmsum,multi_news,\
trec,triviaqa,samsum,\
passage_count,passage_retrieval_en,\
lcc,repobench-p"

mkdir baselines/infllm/benchmark/longbench-result

python baselines/infllm/benchmark/pred.py \
--config_path ${config} \
--output_dir_path baselines/infllm/benchmark/longbench-result \
--datasets ${datasets} 

python baselines/infllm/benchmark/eval.py --dir_path baselines/infllm/benchmark/longbench-result