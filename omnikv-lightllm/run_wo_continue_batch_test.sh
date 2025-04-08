
python lightllm_test.py --num_requests 1 --num_repeat_str 16000 --show_resp  
echo "send request 1"
# sleep 30
python lightllm_test.py --num_requests 2 --num_repeat_str 1000 --show_resp 
echo "send request 2"

# python lightllm_test.py --num_requests 1 --num_repeat_str 1000 --show_resp 
# echo "send request 1"
