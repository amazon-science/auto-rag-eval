from abc import ABC, abstractmethod
from typing import Dict, Generator, Union


class BaseLLM(ABC):

    @abstractmethod
    def invoke(self,
               prompt: str,
               params: Dict[str, Union[int, str]]) -> str:

        pass

    @abstractmethod
    def stream_inference(self,
                         prompt: str,
                         params: Dict[str, Union[int, str]]) -> Generator[str, None, None]:

        pass

    @abstractmethod
    def get_id(self) -> str:

        pass
