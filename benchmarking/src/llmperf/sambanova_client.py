import os
import abc
import sys
import json
import time
import requests
import sseclient
from math import isclose
from datetime import datetime

sys.path.append("./src")
sys.path.append("./src/llmperf")

from transformers import AutoTokenizer
from llmperf.models import RequestConfig
from llmperf import common_metrics
from utils import get_tokenizer

from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

API_BASE_PATH = "/api"

class BaseAPIEndpoint(abc.ABC):
    def __init__(
        self,
        request_config: RequestConfig,
        tokenizer: AutoTokenizer
    ):
        self.request_config = request_config
        self.tokenizer = tokenizer

    @abc.abstractmethod
    def _get_url(self, *args, **kwargs):
        pass
    
    @abc.abstractmethod
    def _get_headers(self, *args, **kwargs):
        pass
    
    @abc.abstractmethod
    def _get_json_data(self, *args, **kwargs):
        pass
    
    def _get_token_length(self, input_text: str) -> int:
        """Gets the token length of a piece of text

    Args:
        request_config (RequestConfig): contains LLM params

    Returns:
        dict: data structure needed for API
    """

    prompt = request_config.prompt_tuple[0]
    
    sampling_params = request_config.sampling_params
    
    if "COE" in request_config.model:
        sampling_params["select_expert"] = request_config.model.split("/")[-1]
        sampling_params["process_prompt"] = False
    
    # generic api v2
    if "/api/v2" in url.lower().strip():  
        tuning_params = json.loads(json.dumps(sampling_params))
        data = {"items": [{"id":"item1", "value": prompt}], "params": tuning_params}
    # support to generic api v1 
    else:   
        tuning_params_dict = {
        k: {"type": type(v).__name__, "value": str(v)}
        for k, v in (sampling_params.items())
        }
        tuning_params = json.dumps(tuning_params_dict)
        
        if request_config.mode == "stream":
            data = {"instance": prompt, "params": json.loads(tuning_params)}
        else:
            data = {"instances": [prompt], "params": json.loads(tuning_params)}
        
    return data


def _compute_client_metrics(
    url: str,
    headers: str,
    request_config: RequestConfig,
    metrics: dict,
    stream: bool,
    tokenizer: AutoTokenizer,
) -> tuple:
    """Gets total time of a request

    Args:
        url (str): URL of the request
        headers (str): headers of the request
        request_config (RequestConfig): request config with parameter info
        metrics (dict): list of metrics with standardized names
        stream (bool): stream option
        tokenizer (AutoTokenizer): tokenizer for counting tokens
    Returns:
        tuple: tuple containing request time, LLM generated text and LLM params
    """

    # Get data
    input_data = _get_data(request_config, url)
    prompt_len = request_config.prompt_tuple[1]

    metrics[common_metrics.REQ_START_TIME] = datetime.now().strftime("%H:%M:%S")
    start_time = chunk_start_time = time.monotonic()
    total_request_time = 0
    ttft = 0
    generated_text = ""
    chunks_received = []
    chunks_timings = []

    # Processing LLM API calls
    with requests.post(
        url, headers=headers, json=input_data, stream=stream
    ) as response:
        if response.status_code != 200:
            response.raise_for_status()
        else:
            if "/api/v2" in url.lower().strip():
                for chunk_orig in response.iter_lines(chunk_size=None):
                    chunk = chunk_orig.strip()
                    data = json.loads(chunk)
                
                    completion = data["result"]["items"][0]["value"]["is_last_response"]
                    chunks_timings.append(time.monotonic() - chunk_start_time)
                    chunk_start_time = time.monotonic()
                    if completion is False:
                        chunks_received.append(data["result"]["items"][0]["value"]["stream_token"])
                        continue
                    else:
                        generated_text = data["result"]["items"][0]["value"]["completion"]
                        response_dict = data["result"]["items"][0]["value"]
                    break
            else: 
                for chunk_orig in response.iter_lines(chunk_size=None):
                    chunk = chunk_orig.strip()
                    data = json.loads(chunk)

                    ##TODO: Non-streaming case
                    if stream is False:
                        generated_text = data["predictions"][0]["completion"]
                        break

                    completion = data["result"]["responses"][0]["is_last_response"]
                    chunks_timings.append(time.monotonic() - chunk_start_time)
                    chunk_start_time = time.monotonic()
                    if completion is False:
                        chunks_received.append(data["result"]["responses"][0]["stream_token"])
                        continue
                    else:
                        generated_text = data["result"]["responses"][0]["completion"]
                        response_dict = data["result"]["responses"][0]
                    break
                
    total_request_time = time.monotonic() - start_time
    metrics[common_metrics.REQ_END_TIME] = datetime.now().strftime("%H:%M:%S")

    # Retrieve server performance metrics if available and calculate client-side metrics
    number_chunks_recieved = len(chunks_received)
    if number_chunks_recieved <= 1:
        ttft = total_request_time
    else:
        Args:
            input_text (str): input text
            
        Returns:
            int: number of tokens
        """
        return len(self.tokenizer.encode(input_text))
    
    
    def _calculate_tpot_from_streams_after_first(self, chunks_received: list, chunks_timings: list) -> float:
        """Calculates Time per Output Token (TPOT) based on the streaming events coming after the first one.
        In general, the way to calculate this metric is: time_to_generate_tokens/number_of_tokens_generated

        Args:
            chunks_received (list): complete list of events coming from streaming response
            chunks_timings (list): complete list of timings that each event took to process

        Returns:
            float: calculated tpot 
        """
        
        # Calculate tokens
        total_tokens_received_after_first_chunk = sum(
            _get_token_length(c, tokenizer) for c in chunks_received[1:]
        )
        total_time_to_receive_tokens_after_first_chunk = sum(chunks_timings[1:])
        total_tokens_in_first_chunk = _get_token_length(chunks_received[0], tokenizer)
        tpot = (
            total_time_to_receive_tokens_after_first_chunk
            / total_tokens_received_after_first_chunk
        )
        ttft = chunks_timings[0] - (total_tokens_in_first_chunk - 1) * tpot

    # Populate server and client metrics
    num_output_tokens = _get_token_length(generated_text, tokenizer)
    metrics = _populate_server_metrics(response_dict, metrics)
    metrics = _populate_client_metrics(
        metrics,
        prompt_len,
        total_request_time,
        ttft,
        num_output_tokens,
        number_chunks_recieved,
    )

    return metrics, generated_text


def _populate_client_metrics(
    metrics: dict,
    prompt_len: int,
    total_request_time: int,
    ttft: int,
    num_output_tokens: int,
    number_chunks_recieved: int,
) -> dict:
    """Populates `metrics` dictionary with performance metrics calculated from client side
        
        return tpot
    
    def _calculate_ttft_from_streams(self, chunks_received: list, chunks_timings: list, total_request_time: int) -> float:
        """Calculates Time to First Token (TTFT) based on the streaming events coming from the response.
        If there are enough streaming events, the formula to calculate ttft is: time_first_chunk - (tokens_first_chunk - 1) * tpot

        Args:
            chunks_received (list): list of events having the streaming tokens
            chunks_timings (list): list of timings for each event
            total_request_time (int): total request time calculated from client side

        Returns:
            float: calculated ttft
        """
        
        number_chunks_recieved = len(chunks_received)
        
        # if one or no chunks were recieved
        if number_chunks_recieved <= 1:
            ttft = total_request_time
        else:
            # calculate tpot
            tpot = self._calculate_tpot_from_streams_after_first(chunks_received, chunks_timings)
            # calculate ttft
            total_tokens_in_first_chunk = self._get_token_length(chunks_received[0])
            ttft = chunks_timings[0] - (total_tokens_in_first_chunk - 1) * tpot  
        return ttft
    
    def _populate_client_metrics(
        self,
        prompt_len: int,
        num_output_tokens: int,
        ttft: int,
        total_request_time: int,
        server_metrics: dict,
        number_chunks_recieved: int,
    ) -> dict:
        """Populates `metrics` dictionary with performance metrics calculated from client side

    Args:
        metrics (dict):  metrics dictionary
        prompt_len (int): prompt's length
        total_request_time (int): end-to-end latency
        ttft (int): time to first token
        num_output_tokens (int): number of output tokens
        number_chunks_recieved (int): number of chunks recieved

    Returns:
        dict: updated metrics dictionary
    """
    metrics[common_metrics.NUM_INPUT_TOKENS] = (
        prompt_len
        if metrics[common_metrics.NUM_INPUT_TOKENS_SERVER] is None
        else metrics[common_metrics.NUM_INPUT_TOKENS_SERVER]
    )
    metrics[common_metrics.NUM_OUTPUT_TOKENS] = (
        num_output_tokens
        if metrics[common_metrics.NUM_OUTPUT_TOKENS_SERVER] is None
        else metrics[common_metrics.NUM_OUTPUT_TOKENS_SERVER]
    )
    metrics[common_metrics.NUM_TOTAL_TOKENS] = (
        prompt_len + num_output_tokens
        if metrics[common_metrics.NUM_TOTAL_TOKENS_SERVER] is None
        else metrics[common_metrics.NUM_TOTAL_TOKENS_SERVER]
    )
    metrics[common_metrics.TTFT] = ttft
    metrics[common_metrics.E2E_LAT] = total_request_time

    if number_chunks_recieved == 1:
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = (
            metrics[common_metrics.NUM_OUTPUT_TOKENS] / total_request_time
        )
    else:
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = (
            metrics[common_metrics.NUM_OUTPUT_TOKENS] / (total_request_time - ttft)
            if not isclose(ttft, total_request_time, abs_tol=1e-8)
            else None
        )
        
    metrics[common_metrics.TOTAL_TOKEN_THROUGHPUT] = (prompt_len + num_output_tokens)/total_request_time
    
    return metrics


def _get_token_length(input_text: str, tokenizer: AutoTokenizer) -> int:
    """Gets the token length of a piece of text

    Args:
        input_text (str): input text
        tokenizer (AutoTokenizer): HuggingFace tokenizer

    Returns:
        int: number of tokens
    """
    return len(tokenizer.encode(input_text))


def _populate_server_metrics(response_dict: dict, metrics: dict) -> dict:
    """Parse output data to metrics dictionary structure
    def _populate_server_metrics(self, response_dict: dict, metrics: dict) -> dict:
        """Parse output data to metrics dictionary structure

    Args:
        output_data (dict): output data with performance metrics
        metrics (dict): metrics dictionary

    Returns:
        dict: updated metrics dictionary
    """
        Returns:
            dict: updated metrics dictionary
        """

    metrics[common_metrics.NUM_INPUT_TOKENS_SERVER] = response_dict.get(
        "prompt_tokens_count"
    )
    metrics[common_metrics.NUM_OUTPUT_TOKENS_SERVER] = response_dict.get(
        "completion_tokens_count"
    )
    metrics[common_metrics.NUM_TOTAL_TOKENS_SERVER] = response_dict.get(
        "total_tokens_count"
    )
    ttft_server = response_dict.get("time_to_first_token") or response_dict.get(
        "time_to_first_response"
    )
    metrics[common_metrics.TTFT_SERVER] = ttft_server
    metrics[common_metrics.E2E_LAT_SERVER] = response_dict.get(
        "total_latency"
    ) or response_dict.get("model_execution_time")
    metrics[common_metrics.REQ_OUTPUT_THROUGHPUT_SERVER] = (
        response_dict.get("completion_tokens_after_first_per_sec")
        or response_dict.get("completion_tokens_per_sec_after_first_response")
        or response_dict.get("throughput_after_first_token")
    )
    metrics[common_metrics.TOTAL_TOKEN_THROUGHPUT_SERVER] = response_dict.get("total_tokens_per_sec") 
    if (metrics[common_metrics.TOTAL_TOKEN_THROUGHPUT_SERVER] is None) and (metrics[common_metrics.E2E_LAT_SERVER] is not None):
        metrics[common_metrics.TOTAL_TOKEN_THROUGHPUT_SERVER] = metrics[common_metrics.NUM_TOTAL_TOKENS_SERVER] / (metrics[common_metrics.E2E_LAT_SERVER])
    metrics[common_metrics.BATCH_SIZE_USED] = response_dict.get("batch_size_used")
    return metrics


=======
>>>>>>> rodrigom/benchmarking_support_fastapi
if __name__ == "__main__":
    # The call of this python file is more for debugging purposes

    # load env variables
    load_dotenv("../.env", override=True)
    env_vars = dict(os.environ)

    # model = "COE/llama-2-7b-chat-hf"
    # model = "COE/llama-2-13b-chat-hf"
    # model = "COE/Mistral-7B-Instruct-v0.2"
    # model = "COE/Meta-Llama-3-8B-Instruct"
    model = "COE/Meta-Llama-3-8B-Instruct"
    llm_api = "sambastudio"
    tokenizer = get_tokenizer(model)

    prompt = "This is a test example, so tell me about anything"
    request_config = RequestConfig(
        prompt_tuple=(prompt, 10),
        model=model,
        llm_api=llm_api,
        sampling_params={
            # "do_sample": False,
            "max_tokens_to_generate": 250,
            # "top_k": 40,
            # "top_p": 0.95,
            # "process_prompt": "False",
        },
        mode="stream",
        num_concurrent_workers=1,
    )

    metrics, generated_text, request_config = llm_request(request_config, tokenizer)

    print(f"Metrics collected: {metrics}")
    # print(f'Completion text: {generated_text}')
    print(f"Request config: {request_config}")
