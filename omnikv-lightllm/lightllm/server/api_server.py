# Adapted from vllm/entrypoints/api_server.py
# of the vllm-project/vllm GitHub repository.
#
# Copyright 2023 ModelTC Team
# Copyright 2023 vLLM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import collections
import time
import torch
import uvloop
import sys
import os
import rpyc
from .build_prompt import build_prompt

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import json
from http import HTTPStatus
import uuid
import multiprocessing as mp
from typing import AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import Response, StreamingResponse, JSONResponse
import uvicorn
from .api_cli import make_argument_parser
from .sampling_params import SamplingParams
from .multimodal_params import MultimodalParams
from .httpserver.manager import HttpServerManager
from .detokenization.manager import start_detokenization_process
from .router.manager import start_router_process
from .embed_cache.manager import start_cache_manager
from .metrics.manager import start_metric_manager
from .visualserver.manager import start_visual_process
from .req_id_generator import ReqIDGenerator
from .api_tgi import tgi_generate_impl, tgi_generate_stream_impl
from .api_lightllm import lightllm_generate, lightllm_generate_stream, lightllm_get_score
from lightllm.utils.net_utils import alloc_can_use_network_port
from lightllm.utils.start_utils import start_submodule_processes

from .api_models import (
    ChatCompletionRequest,
    UsageInfo,
    ChatMessage,
    ChatCompletionResponseChoice,
    ChatCompletionResponse,
    DeltaMessage,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
)

from lightllm.utils.log_utils import init_logger
from prometheus_client import generate_latest
from lightllm.server.metrics.manager import MetricClient

logger = init_logger(__name__)

TIMEOUT_KEEP_ALIVE = 5  # seconds.

g_id_gen = ReqIDGenerator()
app = FastAPI()
server = uvicorn.Server(uvicorn.Config(app))

isFirst = True
metric_client = None
global args
args = None


def first_set_handle_loop():
    global isFirst
    if isFirst:
        loop = asyncio.get_event_loop()
        loop.create_task(httpserver_manager.handle_loop())
        isFirst = False
    return


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    metric_client.counter_inc("lightllm_request_failure")
    return JSONResponse({"message": message}, status_code=status_code.value)


@app.get("/liveness")
@app.post("/liveness")
def liveness():
    return {"status": "ok"}


@app.get("/readiness")
@app.post("/readiness")
def readiness():
    return {"status": "ok"}


@app.get("/get_model_name")
@app.post("/get_model_name")
def get_model_name():
    global args
    return {"model_name": args.model_name}


@app.get("/healthz", summary="Check server health")
@app.get("/health", summary="Check server health")
@app.head("/health", summary="Check server health")
async def healthcheck(request: Request):
    first_set_handle_loop()
    if os.environ.get("DEBUG_HEALTHCHECK_RETURN_FAIL") == "true":
        return JSONResponse({"message": "Error"}, status_code=404)

    from lightllm.utils.health_check import health_check

    if await health_check(httpserver_manager, g_id_gen, request):
        return JSONResponse({"message": "Ok"}, status_code=200)
    else:
        return JSONResponse({"message": "Error"}, status_code=404)


@app.get("/token_load", summary="Get the current server's load of tokens")
async def token_load(request: Request):
    return JSONResponse(
        {
            # 当前使用token量，估计的负载
            "current_load": float(shared_token_load.get_current_load()),
            # 朴素估计的负载，简单将当前请求的输入和输出长度想加得到,目前已未使用，其值与dynamic_max_load一样。
            "logical_max_load": float(shared_token_load.get_logical_max_load()),
            # 动态估计的最大负载，考虑请求中途退出的情况的负载
            "dynamic_max_load": float(shared_token_load.get_dynamic_max_load()),
        },
        status_code=200,
    )


@app.post("/generate")
async def generate(request: Request) -> Response:
    first_set_handle_loop()
    try:
        return await g_generate_func(request, g_id_gen, httpserver_manager)
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@app.post("/generate_stream")
async def generate_stream(request: Request) -> Response:
    first_set_handle_loop()
    try:
        return await g_generate_stream_func(request, g_id_gen, httpserver_manager)
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@app.post("/get_score")
async def get_score(request: Request) -> Response:
    first_set_handle_loop()
    try:
        return await lightllm_get_score(request, g_id_gen, httpserver_manager)
    except Exception as e:
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@app.post("/")
async def compat_generate(request: Request) -> Response:
    request_dict = await request.json()
    stream = request_dict.pop("stream", False)
    if stream:
        return await generate_stream(request)
    else:
        return await generate(request)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, raw_request: Request) -> Response:
    first_set_handle_loop()

    if request.logit_bias is not None:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            "The logit_bias parameter is not currently supported",
        )

    if request.function_call != "none":
        return create_error_response(HTTPStatus.BAD_REQUEST, "The function call feature is not supported")

    created_time = int(time.time())
    prompt = await build_prompt(request)
    sampling_params = SamplingParams(
        do_sample=request.do_sample,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        ignore_eos=request.ignore_eos,
        max_new_tokens=request.max_tokens,
        stop_sequences=request.stop,
        n=request.n,
        best_of=request.n,
    )
    sampling_params.verify()
    multimodal_params = MultimodalParams(images=[])
    multimodal_params.verify_and_preload()

    group_request_id = g_id_gen.generate_id()
    results_generator = httpserver_manager.generate(
        prompt, sampling_params, group_request_id, multimodal_params, request=raw_request
    )

    # Non-streaming case
    if not request.stream:
        final_output_dict = collections.defaultdict(list)
        count_output_tokens_dict = collections.defaultdict(lambda: 0)
        finish_reason_dict = {}
        prompt_tokens_dict = {}
        completion_tokens = 0
        async for sub_req_id, request_output, metadata, finish_status in results_generator:
            count_output_tokens_dict[sub_req_id] += 1
            final_output_dict[sub_req_id].append(request_output)
            if finish_status.is_finished():
                finish_reason_dict[sub_req_id] = finish_status.get_finish_reason()
                prompt_tokens_dict[sub_req_id] = metadata["prompt_tokens"]
        choices = []
        sub_ids = list(final_output_dict.keys())[: request.n]
        for i in range(request.n):
            sub_req_id = sub_ids[i]
            prompt_tokens = prompt_tokens_dict[sub_req_id]
            completion_tokens = count_output_tokens_dict[sub_req_id]
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            chat_message = ChatMessage(role="assistant", content="".join(final_output_dict[sub_req_id]))
            choice = ChatCompletionResponseChoice(
                index=i, message=chat_message, finish_reason=finish_reason_dict[sub_req_id]
            )
            choices.append(choice)
        resp = ChatCompletionResponse(
            id=group_request_id, created=created_time, model=request.model, choices=choices, usage=usage
        )
        return resp

    if sampling_params.n != 1:
        raise Exception("stream api only support n = 1")

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        finish_reason = None
        async for sub_req_id, request_output, metadata, finish_status in results_generator:
            delta_message = DeltaMessage(role="assistant", content=request_output)
            if finish_status.is_finished():
                finish_reason = finish_status.get_finish_reason()
            stream_choice = ChatCompletionStreamResponseChoice(
                index=0, delta=delta_message, finish_reason=finish_reason
            )
            stream_resp = ChatCompletionStreamResponse(
                id=group_request_id,
                created=created_time,
                model=request.model,
                choices=[stream_choice],
            )
            yield ("data: " + stream_resp.json(ensure_ascii=False) + "\n\n").encode("utf-8")

    async def abort_request() -> None:
        await httpserver_manager.abort(group_request_id)

    background_tasks = BackgroundTasks()
    # Abort the request if the client disconnects.
    background_tasks.add_task(abort_request)
    return StreamingResponse(stream_results(), media_type="text/event-stream", background=background_tasks)


@app.get("/tokens")
@app.post("/tokens")
async def tokens(request: Request):
    try:
        request_dict = await request.json()
        prompt = request_dict.pop("text")
        return JSONResponse({"ntokens": httpserver_manager.tokens(prompt)}, status_code=200)
    except Exception as e:
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, f"error: {str(e)}")


@app.get("/metrics")
async def metrics() -> Response:
    data = await metric_client.generate_latest()
    response = Response(data)
    response.mimetype = "text/plain"
    return response


@app.on_event("shutdown")
async def shutdown():
    logger.info("Received signal to shutdown. Performing graceful shutdown...")
    await asyncio.sleep(3)
    logger.info("Graceful shutdown completed.")

    # 杀掉所有子进程
    import psutil
    import signal

    parent = psutil.Process(os.getpid())
    children = parent.children(recursive=True)
    for child in children:
        os.kill(child.pid, signal.SIGKILL)

    server.should_exit = True
    return


def main():
    parser = make_argument_parser()
    global args
    args = parser.parse_args()

    global g_generate_func
    global g_generate_stream_func
    if args.use_tgi_api:
        g_generate_func = tgi_generate_impl
        g_generate_stream_func = tgi_generate_stream_impl
    else:
        g_generate_func = lightllm_generate
        g_generate_stream_func = lightllm_generate_stream

    logger.info(f"use tgi api: {args.use_tgi_api}")

    assert args.max_req_input_len < args.max_req_total_len
    assert not (args.beam_mode and args.use_dynamic_prompt_cache), "Beam mode incompatible with dynamic prompt cache"
    assert (
        args.mem_fraction > 0 and args.mem_fraction < 1
    ), f"Invalid mem_fraction {args.mem_fraction}, The expected value is between 0 and 1."

    # splitfuse_mode 和 cuda_graph 不能同时开启
    if args.splitfuse_mode:
        assert args.disable_cudagraph

    # 这些模式不能同时设置。
    assert [
        args.splitfuse_mode,
        args.beam_mode,
        args.diverse_mode,
        args.token_healing_mode,
        args.use_reward_model,
        args.return_all_prompt_logprobs,
        args.first_token_constraint_mode,
    ].count(True) <= 1
    # 部分模式目前还无法与dynamic_prompt_cache一起跑，to do。
    if args.use_dynamic_prompt_cache:
        assert args.beam_mode is False
        assert args.token_healing_mode is False

    # 部分模式还不能支持与高级动态调度算法协同，to do.
    if args.beam_mode or args.diverse_mode:
        assert args.router_token_ratio == 0.0

    # 检查GPU数量是否足够
    total_required_gpus = args.visual_dp * args.visual_tp
    if len(args.visual_gpu_ids) < total_required_gpus:
        raise ValueError(
            f"Not enough GPUs specified. You need at least {total_required_gpus}, but got {len(args.visual_gpu_ids)}."
        )
    else:
        args.visual_gpu_ids = args.visual_gpu_ids[:total_required_gpus]

    # 检查visual_nccl_port数量是否足够
    if len(args.visual_nccl_ports) < args.visual_dp:
        raise ValueError(
            f"Not enough visual_nccl_ports specified. You need at least {args.visual_dp}, "
            f"but got ({len(args.visual_nccl_ports)})."
        )
    else:
        args.visual_nccl_ports = args.visual_nccl_ports[: args.visual_dp]

    if not args.splitfuse_mode:
        # 普通模式下
        if args.batch_max_tokens is None:
            args.batch_max_tokens = args.max_req_total_len
        else:
            assert args.batch_max_tokens >= args.max_req_total_len, "batch_max_tokens must >= max_req_total_len"
    else:
        # splitfuse 模式下
        # assert args.batch_max_tokens is not None, "need to set by yourself"
        if args.batch_max_tokens is None:
            args.batch_max_tokens = min(args.max_req_total_len, 16 * args.splitfuse_block_size)

        assert (
            args.batch_max_tokens > args.splitfuse_block_size
        ), "splitfuse_mode, batch_max_tokens must >= splitfuse_block_size"

    # help to manage data stored on Ceph
    if "s3://" in args.model_dir:
        from lightllm.utils.petrel_helper import s3_model_prepare

        s3_model_prepare(args.model_dir)

    # 如果args.eos_id 是 None, 从 config.json 中读取 eos_token_id 相关的信息，赋值给 args
    if args.eos_id is None:
        from lightllm.utils.config_utils import get_eos_token_ids

        args.eos_id = get_eos_token_ids(args.model_dir)

    if args.data_type is None:
        from lightllm.utils.config_utils import get_dtype

        args.data_type = get_dtype(args.model_dir)
        assert args.data_type in ["fp16", "float16", "bf16", "bfloat16", "fp32", "float32"]

    logger.info(f"all start args:{args}")

    already_uesd_ports = args.visual_nccl_ports + [args.nccl_port]
    can_use_ports = alloc_can_use_network_port(
        num=6 + args.tp + args.visual_dp * args.visual_tp, used_nccl_ports=already_uesd_ports
    )
    router_port, detokenization_port, httpserver_port, visual_port, cache_port, metric_port = can_use_ports[0:6]
    model_rpc_ports = can_use_ports[6 : 6 + args.tp]
    can_use_ports = can_use_ports[6 + args.tp :]

    visual_model_tp_ports = []
    for _ in range(args.visual_dp):
        tp_ports_for_dp = can_use_ports[0 : args.visual_tp]
        can_use_ports = can_use_ports[args.visual_tp :]
        visual_model_tp_ports.append(tp_ports_for_dp)

    if args.enable_multimodal:
        start_submodule_processes(
            start_funcs=[
                start_cache_manager,
            ],
            start_args=[(cache_port, args)],
        )
        start_submodule_processes(
            start_funcs=[
                start_visual_process,
            ],
            start_args=[
                (args, router_port, visual_port, cache_port, visual_model_tp_ports),
            ],
        )

    start_submodule_processes(
        start_funcs=[
            start_metric_manager,
        ],
        start_args=[(metric_port, args)],
    )
    global metric_client
    metric_client = MetricClient(metric_port)

    global httpserver_manager
    httpserver_manager = HttpServerManager(
        args,
        router_port=router_port,
        cache_port=cache_port,
        visual_port=visual_port,
        httpserver_port=httpserver_port,
        enable_multimodal=args.enable_multimodal,
        metric_port=metric_port,
    )

    start_submodule_processes(
        start_funcs=[start_router_process, start_detokenization_process],
        start_args=[
            (args, router_port, detokenization_port, model_rpc_ports, metric_port),
            (args, detokenization_port, httpserver_port),
        ],
    )
    if "s3://" in args.model_dir:
        from lightllm.utils.petrel_helper import s3_model_clear

        s3_model_clear(args.model_dir)

    if args.health_monitor:
        from lightllm.server.health_monitor.manager import start_health_check_process

        start_submodule_processes(start_funcs=[start_health_check_process], start_args=[(args,)])

    # 共享变量，用于获取router端调度分析得到的机器负载信息
    from lightllm.server import TokenLoad

    global shared_token_load
    shared_token_load = TokenLoad(f"{str(args.nccl_port)}_shared_token_load")

    server.install_signal_handlers()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        loop="uvloop",
    )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn"),  # this code will not be ok for settings to fork to subprocess
    main()
