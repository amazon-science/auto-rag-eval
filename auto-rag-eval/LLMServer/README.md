# Large Language Models - Exam Generation

## Provided Models

This folder contains the code for Large Language models to be used for exam Generation. So far, we provide the implementation for Bedrock Claude family of models and will be adding support for more models soon.

Moreover, in `llm_exam_generator.py`, we provide the wrapper on top of `BaseLLM` to get `LLMExamGenerator` class and `ClaudeExamGenerator`

## Custom LLM

The only piece required to add you own LLM is to follow the BaseLLM class from `base_model.py`:

```[python]
class BaseLLM:

    def invoke(self,
               prompt: str) -> str:

        pass

    def stream_inference(self,
                         prompt: str) -> Generator[str, None, None]:

        pass

    def get_id(self) -> str:

        pass
```
