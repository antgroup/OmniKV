import asyncio
import aiohttp
import time
import fire

# 服务的URL
SERVICE_URL = "http://localhost:8100/v1/completions"


# 异步函数，用于发送请求并记录响应时间
async def send_request(session, url, json_data):
    try:
        start_time = time.time()
        async with session.post(url, json=json_data) as response:
            end_time = time.time()
            response_time = end_time - start_time
            status = response.status
            response_text = await response.text()
            return status, response_time, response_text
    except Exception as e:
        return None, None, str(e)


# 测试服务的性能
async def test_service(service_url, service_data, num_requests):
    timeout = aiohttp.ClientTimeout(total=10000)  # 设置总超时时间为1000秒
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for _ in range(num_requests):
            task = send_request(session, service_url, service_data)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return results


# 分析结果
def analyze_results(service_name, results, show_resp):
    total_time = 0
    successful_requests = 0
    failed_requests = 0

    for status, response_time, response_text in results:
        if status == 200:
            successful_requests += 1
            total_time += response_time
        else:
            failed_requests += 1
            print(f"{service_name} - Failed request: {response_text}")

    avg_response_time = (
        total_time / successful_requests if successful_requests > 0 else float("inf")
    )

    print(f"\n{service_name} Results:")
    print(f"Total requests: {len(results)}")
    print(f"Successful requests: {successful_requests}")
    print(f"Failed requests: {failed_requests}")
    print(f"Average response time: {avg_response_time:.2f} seconds")
    if show_resp:
        print(results)


# 运行测试
async def run_tests(num_requests, num_repeat_str, show_resp, max_new_tokens):
    # 动态设置数据
    prompt = "What is AI? " * num_repeat_str
    service_data = {
        "model": "gpt-3.5-turbo",
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": 0,
    }

    print(f"\nTesting Service with {num_repeat_str} repeats of the prompt...")
    results_service = await test_service(SERVICE_URL, service_data, num_requests)
    analyze_results("Service", results_service, show_resp)


# 主函数入口，使用fire处理命令行参数
def main(num_requests=1, num_repeat_str=1500, show_resp=False, max_new_tokens=6000):
    asyncio.run(run_tests(num_requests, num_repeat_str, show_resp, max_new_tokens))


if __name__ == "__main__":
    fire.Fire(main)
