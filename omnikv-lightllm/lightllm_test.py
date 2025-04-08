import asyncio
import aiohttp
import time
import fire
import random

# 服务2的URL
SERVICE_2_URL = "http://localhost:8000/generate"
with open("./harry_potter.txt", "r", encoding="utf-8") as _in:
    data = _in.readlines()
    # data = "\n".join(data)


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
    timeout = aiohttp.ClientTimeout(total=1000)  # 设置总超时时间为1000秒
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for _ in range(num_requests):
            if service_data == "prefix":
                if _ == 0:
                    this_data = "\n".join(data)
                    print(f"request char num {len(this_data)}")
                    gogo = {
                        "inputs": f"{this_data} ...... \n\nPlease summarize this book.",
                        "parameters": {"max_new_tokens": 1000, "frequency_penalty": 0},
                    }
                else:
                    # 动态设置数据
                    ratio = (random.random() * 8 + 2) / 10  # 0.2-1.0
                    this_data = "\n".join(data[: int(len(data) * ratio)])
                    print(f"request char num {len(this_data)}")
                    gogo = {
                        "inputs": f"{this_data} ...... \n\nPlease summarize this book.",
                        "parameters": {"max_new_tokens": 1000, "frequency_penalty": 0},
                    }

            elif service_data == "continue":
                if _ == 0:
                    gogo = {
                        "inputs": "What is AI? " * 16000,
                        "parameters": {"max_new_tokens": 1000, "frequency_penalty": 0},
                    }
                else:
                    gogo = {
                        "inputs": "What is AI? " * 100,
                        "parameters": {"max_new_tokens": 1000, "frequency_penalty": 0},
                    }

            else:
                gogo = service_data

            task = send_request(session, service_url, gogo)
            tasks.append(task)
            # if service_data == "prefix" or service_data == "continue":
            #     print("sleep for first full prompt end")
            #     time.sleep(240)

        results = await asyncio.gather(*tasks)
        return results


# 分析结果
def analyze_results(service_name, results, show_resp):
    total_time = 0
    successful_requests = 0
    failed_requests = 0
    times = []

    for status, response_time, response_text in results:
        if status == 200:
            successful_requests += 1
            total_time += response_time
            times += [response_time]
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
    print(f"Response time: {times} seconds")
    if show_resp:
        print(results)


# 运行测试
async def run_tests_prefix(num_requests, show_resp):
    print(f"\nTesting Service 2 with {num_requests} repeats...")
    results_service_2 = await test_service(SERVICE_2_URL, "prefix", num_requests)
    analyze_results("Service 2 with prefix", results_service_2, show_resp)


async def run_tests_continue(num_requests, show_resp):
    print(f"\nTesting Service whether continue batch")
    results_service_2 = await test_service(SERVICE_2_URL, "continue", num_requests)
    analyze_results("Service 2 with continue", results_service_2, show_resp)


async def run_tests(num_requests, num_repeat_str, show_resp):
    service_data = {
        "inputs": "What is AI? " * num_repeat_str,
        "parameters": {
            "max_new_tokens": 1002,
            "min_new_tokens": 1000,
            "frequency_penalty": 0,
        },
    }

    print(f"\nTesting Service 2 with {num_requests} repeats...")
    results_service_2 = await test_service(SERVICE_2_URL, service_data, num_requests)
    analyze_results("Service 2", results_service_2, show_resp)


# 主函数入口，使用fire处理命令行参数
def main(
    num_requests=1,
    num_repeat_str=1500,
    show_resp=False,
    test_prefix=False,
    test_continue=False,
    seed=42,
):
    random.seed(seed)
    if test_prefix:
        asyncio.run(run_tests_prefix(num_requests, show_resp))
        return
    if test_continue:
        asyncio.run(run_tests_continue(num_requests, show_resp))
        return
    asyncio.run(run_tests(num_requests, num_repeat_str, show_resp))


if __name__ == "__main__":
    fire.Fire(main)
