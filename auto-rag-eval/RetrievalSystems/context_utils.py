import datetime
import inspect
import os
import re
from typing import Dict, List, Union

from pydantic import BaseModel


def string_to_date(string: str):
    year, month, day = map(int, string.split("-"))
    return datetime.date(year, month, day)


def filter_args(func, args_dict):
    sig = inspect.signature(func)
    return {k: v for k, v in args_dict.items() if k in sig.parameters}


class ContextPassage(BaseModel):
    source: Union[str, List[str]]
    docs_id: str
    title: str
    section: Union[str, List[str]]
    text: str
    start_character: Union[str, int]
    end_character: Union[str, int]
    date: str
    answer_similarity: float = 0


class ConstraintException(Exception):
    pass


class ContextProvider:

    def get_context_from_query(self,
                               query: str,
                               params: Dict[str, Union[int, str]] = {}) -> List[ContextPassage]:

        pass


def get_single_file_in_folder(folder_path):
    # List all entries in the given folder
    entries = os.listdir(folder_path)

    # Filter out only the files (excluding directories and other types)
    files = [os.path.join(folder_path, f) for f in entries if os.path.isfile(os.path.join(folder_path, f))]

    # Check the number of files
    if len(files) == 1:
        return files[0]
    elif len(files) == 0:
        raise ValueError(f"No files found in the directory {folder_path}")
    else:
        raise ValueError(f"More than one file found in the directory {folder_path}. Files are: {', '.join(files)}")


def clean_question(text):
    result = cleanup_references(text)
    result = result.replace("\n", " ")
    result = re.sub(r"\s\s+", " ", result)
    result = result.replace("[deleted]", "")
    return result.lower().strip()


def cleanup_references(text):
    # URL reference where we need to remove both the link text and URL
    # ...and this letter is used by most biographers as the cornerstone of Lee's personal
    # views on slavery ([1](_URL_2_ & pg=PA173), [2](_URL_1_), [3](_URL_5_)).
    # ...and this letter is used by most biographers as the cornerstone of Lee's personal views on slavery.
    result = re.sub(r"[\(\s]*\[\d+\]\([^)]+\)[,)]*", "", text, 0, re.MULTILINE)

    # URL reference where we need to preserve link text but remove URL
    # At the outbreak of the Civil War, [Leyburn left his church](_URL_19_) and joined the South.
    # At the outbreak of the Civil War, Leyburn left his church and joined the South.
    result = re.sub(r"\[([^]]+)\]\([^)]+\)", "\\1", result, 0, re.MULTILINE)

    # lastly remove just dangling _URL_[0-9]_ URL references
    result = re.sub(r"_URL_\d_", "", result, 0, re.MULTILINE)
    return result


def clean_answer(text):
    result = cleanup_references(text)
    result = result.replace("\n", " ")
    result = re.sub(r"\s\s+", " ", result)
    result = re.sub(r"BULLET::::-", "", result)
    return trim(result.strip())


def trim(text, word_count: int = 100):
    return " ".join(text.split(" ")[:word_count])
