import json
import logging
from typing import Dict, Generator, List, Literal

import boto3
from botocore.config import Config
from LLMServer.base_model import BaseLLM
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

STOP_AFTER_ATTEMPT = 6
WAIT_EXPONENTIAL_MIN = 4
WAIT_EXPONENTIAL_MAX = 30

AllowedModelNames = Literal['anthropic.claude-3-sonnet-20240229-v1:0',
                            'anthropic.claude-3-haiku-20240307-v1:0']


class ClaudeV3(BaseLLM):

    def __init__(self,
                 model_name: AllowedModelNames):

        self.model_name = model_name
        self.inference_params = {
            # max_tokens_to_sample can be at most 4096
            "max_tokens_to_sample": 4096,
            "temperature": 0,
            "top_p": 0.9,
        }
        self.brt = boto3.client("bedrock-runtime",
                                region_name="us-east-1",
                                config=Config(read_timeout=1000)
                                )

        # To bypass the length limit, we set a maximum number of iterations
        self.max_n_iterations = 20

    @retry(
        stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
        wait=wait_exponential(min=WAIT_EXPONENTIAL_MIN,
                              max=WAIT_EXPONENTIAL_MAX),
    )
    def _get_bedrock_response(self, query: str) -> Dict:

        messages = self.query_to_messages(query)
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.inference_params['max_tokens_to_sample'],
            "messages": messages,
            "temperature": self.inference_params['temperature'],
        }
        response = self.brt.invoke_model(
            modelId=self.model_name,
            body=json.dumps(request_body),
        )
        result = json.loads(response.get('body').read())
        return result

    def invoke(self,
               prompt: str) -> str:

        result = self._get_bedrock_response(
            query=prompt)

        return correct_generation(
            result["content"][0]["text"])

    def invoke_with_prolongation(self,
                                 prompt: str) -> str:
        '''
        Claude Bedrock strickly enforces a lenght of 4096 tokens.
        This method will try to prolong the output until the model
        reaches the desired output by feeding back the generated text
        into the prompt, at most self.max_n_iterations.
        '''

        generated_text = ""
        generated_text_with_query = prompt + "\nAssistant: "

        for i in range(self.max_n_iterations):

            response = self._get_bedrock_response(
                query=generated_text_with_query)
            generated_chunk = correct_generation(
                response["content"][0]["text"])

            # TODO: Ensure that strip usage is relevant
            generated_text += generated_chunk.strip()
            generated_text_with_query += generated_chunk.strip()

            logger.info(f"Prolongation of generation at round {i}")

            if not self._has_enforced_interruption(response=response):
                return generated_text

        raise ValueError("Failure to complete prompt: File is too big.")

    def _has_enforced_interruption(self,
                                   response: Dict) -> bool:
        '''
        Detect if the generation ends because of end of text or
        forced interuptions.
        '''
        if response.get("stop_reason", "") == "max_tokens":
            return True

        token_delta = 10
        # Check if the model has reached the maximum token limit
        if response.get("usage", {}).get('output_tokens', 0) >= self.inference_params['max_tokens_to_sample'] - token_delta:
            return True

        return False

    def query_to_messages(query: str) -> List[Dict[str, str]]:
        human_tag = "Human:"
        assistant_tag = "Assistant:"
        if query.startswith(human_tag):
            query = query[len(human_tag):]
        if assistant_tag in query:
            query, prefill = query.split(assistant_tag, 1)
            return [
                {"role": "user", "content": query},
                {"role": "assistant", "content": prefill.strip()},
            ]
        return [{"role": "user", "content": query}]

    def stream_inference(self,
                         prompt: str) -> Generator[str, None, None]:
        pass

    def get_id(self) -> str:
        return self.model_name


def correct_generation(generated_text: str) -> str:

    return generated_text.replace('&lt;', '<').replace('&gt;', '>')
