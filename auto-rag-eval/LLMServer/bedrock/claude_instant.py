import json
import time
from typing import Generator

import boto3
from botocore.config import Config
from LLMServer.base_model import BaseLLM
from tenacity import retry, stop_after_attempt, wait_exponential

STOP_AFTER_ATTEMPT = 6
WAIT_EXPONENTIAL_MIN = 4
WAIT_EXPONENTIAL_MAX = 30


def delayed_text_generator(text: str, delay: float = 0.2):
    tokens = text.split()
    for i in range(1, len(tokens) + 1):
        time.sleep(delay)
        yield ' '.join(tokens[:i])


class ClaudeInstant(BaseLLM):

    def __init__(self):
        self.bedrock = boto3.client(
            service_name='bedrock',
            config=Config(read_timeout=1000))
        self.modelId = 'anthropic.claude-instant-v1'
        self.accept = 'application/json'
        self.contentType = 'application/json'
        self.inference_params = {
            # max_tokens_to_sample can be at most 4096
            "max_tokens_to_sample": 4096,
            "temperature": 0,
            "top_p": 0.9,
        }

    @retry(
        stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
        wait=wait_exponential(min=WAIT_EXPONENTIAL_MIN,
                              max=WAIT_EXPONENTIAL_MAX),
    )
    def invoke(self,
               prompt: str) -> str:

        body = json.dumps({
            "prompt": prompt,
            **self.inference_params
        })

        response = self.bedrock.invoke_model(body=body,
                                             modelId=self.modelId,
                                             accept=self.accept,
                                             contentType=self.contentType)
        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            return json.loads(
                response.get('body').read()).get('completion')

        raise ValueError("Incorrect Generation")

    @retry(
        stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
        wait=wait_exponential(min=WAIT_EXPONENTIAL_MIN,
                              max=WAIT_EXPONENTIAL_MAX),
    )
    def stream_inference(self,
                         prompt: str) -> Generator[str, None, None]:

        body = json.dumps({
            "prompt": prompt,
            **self.inference_params
        })

        response = self.bedrock.invoke_model(body=body,
                                             modelId=self.modelId,
                                             accept=self.accept,
                                             contentType=self.contentType)
        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            return delayed_text_generator(json.loads(
                response.get('body').read()).get('completion'))

        raise ValueError("Incorrect Generation")

    def get_id(self):

        return "ClaudeV2:Instant"
