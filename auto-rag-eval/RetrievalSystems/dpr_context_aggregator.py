import logging
from os.path import abspath, dirname
from typing import Dict, List, Union

import numpy as np
from RetrievalSystems.context_utils import (
    ContextPassage,
    ContextProvider,
    SearchConstraint,
    filter_args,
)
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

ROOTPATH = dirname(dirname(abspath(__file__)))


class DPRContextGenerator(ContextProvider):

    # Here the SearchAggregator Works over a single ContextProvider class
    TOPK_CROSSENCODER = 3

    def __init__(self,
                 context_sources: Dict[str, ContextProvider]):

        self.context_sources = context_sources
        self.crossencoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.search_constraints = SearchConstraint()

    def get_matching_context(self,
                             query: str) -> List[ContextPassage]:

        context_passages = []

        # TODO: Run in parallel
        for context_provider_id, context_provider in self.context_sources.items():

            try:

                context_passages.extend(
                    context_provider.get_context_from_query(query=query))
                # logger.info(
                #    f'{context_provider_id} Context successfully extracted for query "{query}"')

            except Exception as e:
                logger.error(
                    f'Failure to extract {context_provider_id} context for query "{query}": {e}')

        return context_passages

    def get_ranked_context(self,
                           query: str,
                           context_passages: List[ContextPassage],
                           topk_crossencoder: int = TOPK_CROSSENCODER) -> List[ContextPassage]:

        question_passage_combinations = [
            [query, p.text] for p in context_passages]

        # Compute the similarity scores for these combinations
        similarity_scores = self.crossencoder.predict(
            question_passage_combinations)

        # Sort the scores in decreasing order
        sim_ranking_idx = np.flip(np.argsort(similarity_scores))

        return [context_passages[rank_idx]
                for rank_idx in sim_ranking_idx[:topk_crossencoder]]

    def get_context_from_query(self,
                               query: str,
                               params: Dict[str, Union[int, str]] = {}) -> List[ContextPassage]:

        preprocessed_query = query.replace('\n', ' ')
        context_passages = self.get_matching_context(
            query=preprocessed_query)

        ranked_context_passages = self.get_ranked_context(
            query=preprocessed_query,
            context_passages=context_passages,
            **filter_args(func=self.get_ranked_context,
                          args_dict=params))

        return ranked_context_passages
