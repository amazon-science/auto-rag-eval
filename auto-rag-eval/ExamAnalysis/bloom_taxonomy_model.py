from typing import Dict, List

import spacy

# If required, download the spaCy English model
# !python -m spacy download en_core_web_sm

TAXONOMY_V2 = {'Remembering': ['list', 'identify', 'name', 'define', 'mention', 'recall', 'label', 'state', 'recognize', 'repeat'],
               'Understanding': ['explain', 'describe', 'summarize', 'predict', 'interpret', 'paraphrase', 'translate', 'illustrate', 'rephrase', 'clarify', 'check', 'find', 'experience', 'suspect', 'review', 'notice', 'assume', 'interact', 'observe', 'understand'],
               'Applying': ['demonstrate', 'apply', 'use', 'illustrate', 'solve', 'show', 'execute', 'implement', 'operate', 'practice', 'set', 'configure', 'use', 'try', 'follow', 'take', 'use', 'run', 'serve', 'task', 'operate', 'work', 'enable', 'exist', 'read', 'write'],
               'Analyzing': ['analyze', 'distinguish', 'compare', 'differentiate', 'examine', 'test', 'question', 'inspect', 'debate', 'investigate', 'manage', 'resolve', 'optimize', 'troubleshoot', 'investigate', 'compare', 'differentiate'],
               'Evaluating': ['evaluate', 'rate', 'justify', 'critique', 'decide', 'rank', 'measure', 'recommend', 'test', 'validate', 'assess', 'evaluate', 'decide', 'choose', 'verify', 'test', 'monitor', 'validate', 'recommend'],
               'Creating': ['design', 'construct', 'produce', 'invent', 'devise', 'formulate', 'originate', 'assemble', 'generate', 'compose', 'create', 'design', 'develop', 'generate', 'implement', 'produce', 'build', 'customize', 'formulate']}

TAXONOMY_V1 = {"Remembering" : ["list", "identify", "name", "define", "mention", "recall", "label", "state", "recognize", "repeat"],
               "Understanding" : ["explain", "describe", "summarize", "predict", "interpret", "paraphrase", "translate", "illustrate", "rephrase", "clarify"],
               "Applying" : ["demonstrate", "apply", "use", "illustrate", "solve", "show", "execute", "implement", "operate", "practice"],
               "Analyzing" : ["analyze", "distinguish", "compare", "differentiate", "examine", "test", "question", "inspect", "debate", "investigate"],
               "Evaluating" : ["evaluate", "rate", "justify", "critique", "decide", "rank", "measure", "recommend", "test", "validate", "assess"],
               "Creating" : ["design", "construct", "produce", "invent", "devise", "formulate", "originate", "assemble", "generate", "compose"]}


def categorize_question(question: str,
                        taxonomy: Dict[str, List[str]] = TAXONOMY_V2) -> List[str]:
    '''
    Categorize questions using Bloom's taxonomy approximation.

    Parameters:
    question (str): The question to categorize
    taxonomy (Dict[str, List[str]]): The taxonomy to use for categorization
    '''

    nlp = spacy.load("en_core_web_sm")

    # Define verb lists for each category

    # Convert the question to lowercase and split it into words
    doc = nlp(question)
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]

    # Check for verbs from each category
    classif = [key
               for key in taxonomy.keys()
               if any(verb in verbs for verb in taxonomy[key])]

    return classif if len(classif) > 0 else ["Uncategorized"]
