"""Wrapper around Sambanova fast CoE API."""

import json
from typing import Any, Dict, Generator, Iterator, List, Optional, Sequence, Union

import requests
from langchain.callbacks.base import BaseCallbackHandler  # type: ignore
from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    BaseCallbackManager,
    CallbackManager,
    CallbackManagerForLLMRun,
    Callbacks,
)
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel, LanguageModelInput
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult, RunInfo
from langchain_core.prompt_values import ChatPromptValue, PromptValue, StringPromptValue
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env


class SambaStudioFastCoE(LLM):
    """
    SambaStudio large language models.

    To use, you should have the environment variables
    ``FAST_COE_URL`` set with your SambaStudio environment URL.
    ``FAST_COE_API_KEY``  set with your SambaStudio endpoint API key.

    https://sambanova.ai/products/enterprise-ai-platform-sambanova-suite

    read extra documentation in https://docs.sambanova.ai/sambastudio/latest/index.html

    Example:
    .. code-block:: python

        from langchain_community.llms.sambanova  import Sambaverse
        SambaStudio(
            fast_coe_url="your fast CoE endpoint URL",
            api_token= set with your fast CoE endpoint API key.,
            max_tokens = mas number of tokens to generate
            stop_tokens = list of stop tokens
            model = model name
        )
    """

    fast_coe_url: str = ''
    """Url to use"""

    fast_coe_api_key: str = ''
    """fastCoE api key"""

    max_tokens: int = 1024
    """max tokens to generate"""

    stop_tokens: list = ['<|eot_id|>']
    """Stop tokens"""

    model: str = 'llama3-8b'
    """LLM model expert to use"""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {'model': self.model, 'max_tokens': self.max_tokens, 'stop': self.stop_tokens}

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return 'Sambastudio Fast CoE'

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values['fast_coe_url'] = get_from_dict_or_env(values, 'fast_coe_url', 'FAST_COE_URL')
        values['fast_coe_api_key'] = get_from_dict_or_env(values, 'fast_coe_api_key', 'FAST_COE_API_KEY')
        return values

    def _handle_nlp_predict_stream(
        self,
        prompt: Union[List[str], str],
        stop: List[str],
    ) -> Iterator[GenerationChunk]:
        """
        Perform a streaming request to the LLM.

        Args:
            prompt: The prompt to use for the prediction.
            stop: list of stop tokens

        Returns:
            An iterator of GenerationChunks.
        """
        try:
            import sseclient
        except ImportError:
            raise ImportError('could not import sseclient library' 'Please install it with `pip install sseclient-py`.')
        try:
            formatted_prompt = json.loads(prompt)
        except:
            formatted_prompt = [{'role': 'user', 'content': prompt}]

        http_session = requests.Session()
        if not stop:
            stop = self.stop_tokens
        data = {'inputs': formatted_prompt, 'max_tokens': self.max_tokens, 'stop': stop, 'model': self.model}

        # Streaming output
        response = http_session.post(
            self.fast_coe_url,
            headers={'Authorization': f'Basic {self.fast_coe_api_key}', 'Content-Type': 'application/json'},
            json=data,
            stream=True,
        )

        client = sseclient.SSEClient(response)
        close_conn = False
        for event in client.events():
            if event.event == 'error_event':
                close_conn = True
            chunk = {
                'event': event.event,
                'data': event.data,
                'status_code': response.status_code,
            }

            if chunk['status_code'] == 200 and chunk.get('error'):
                chunk['result'] = {'responses': [{'stream_token': ''}]}
            if chunk['status_code'] != 200:
                error = chunk.get('is_error')
                optional_details = chunk.get('completion')
                if error:
                    optional_details = error.get('details')
                    raise ValueError(
                        f"Sambanova /complete call failed with status code "
                        f"{chunk['status_code']}.\n"
                        f"Details: {optional_details}\n"
                    )
                else:
                    raise RuntimeError(
                        f"Sambanova /complete call failed with status code " f"{chunk['status_code']}." f"{chunk}."
                    )
            text = json.loads(chunk['data'])['stream_token']
            generated_chunk = GenerationChunk(text=text)
            yield generated_chunk

    def _stream(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Call out to Sambanova's complete endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.
        """
        try:
            for chunk in self._handle_nlp_predict_stream(prompt, stop):
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text)
                yield chunk
        except Exception as e:
            # Handle any errors raised by the inference endpoint
            raise ValueError(f'Error raised by the inference endpoint: {e}') from e

    def _handle_stream_request(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]],
        run_manager: Optional[CallbackManagerForLLMRun],
        kwargs: Dict[str, Any],
    ) -> str:
        """
        Perform a streaming request to the LLM.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
            run_manager: Callback manager for the run.
            **kwargs: Additional keyword arguments. directly passed
                to the sambaverse model in API call.

        Returns:
            The model output as a string.
        """
        completion = ''
        for chunk in self._stream(prompt=prompt, stop=stop, run_manager=run_manager, **kwargs):
            completion += chunk.text
        return completion

    def _call(
        self,
        prompt: Union[List[str], str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to Sambanova's complete endpoint.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.
        """
        try:
            return self._handle_stream_request(prompt, stop, run_manager, kwargs)
        except Exception as e:
            # Handle any errors raised by the inference endpoint
            raise ValueError(f'Error raised by the inference endpoint: {e}') from e
