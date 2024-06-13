def make_prompt_closed_book(doc):

    return "###Human: Question: {}\n\nCandidates:\n{}\n\n###Assistant: Correct answer".format(doc['question'],
                                                                                              "\n".join(doc['choices']))


def make_prompt_open_book(doc):

    return "###Human: Question: {}\n\nContext:{}\n\nCandidates:\n{}\n\n###Assistant: Correct answer".format(doc['question'],
                                                                                                            doc['documentation'],
                                                                                                            "\n".join(doc['choices']))


def make_prompt_rag_dpr(doc, n_retrieved_docs: int = 1):

    return "###Human: Question: {}\n\Retrieved Documents:\n{}\n\nCandidates:\n{}\n\n###Assistant: Correct answer".format(doc['question'],
                                                                                                                         "\n".join(
                                                                                                                             doc['retrieved_context']['DPR'][:n_retrieved_docs]),
                                                                                                                         "\n".join(doc['choices']))


def make_prompt_rag_siamese(doc, n_retrieved_docs: int = 1):

    return "###Human: Question: {}\n\Retrieved Documents:\n{}\n\nCandidates:\n{}\n\n###Assistant: Correct answer".format(doc['question'],
                                                                                                                         "\n".join(
                                                                                                                             doc['retrieved_context']['SIAMESE'][:n_retrieved_docs]),
                                                                                                                         "\n".join(doc['choices']))


def make_prompt_rag_bm25(doc, n_retrieved_docs: int = 1):

    return "###Human: Question: {}\n\Retrieved Documents:\n{}\n\nCandidates:\n{}\n\n###Assistant: Correct answer".format(doc['question'],
                                                                                                                         "\n".join(
                                                                                                                             doc['retrieved_context']['BM25'][:n_retrieved_docs]),
                                                                                                                         "\n".join(doc['choices']))
