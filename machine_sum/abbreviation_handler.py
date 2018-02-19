#
#   Handle abbreviations
#   Simple parser, ~70% sensitivity
#   Does not work for cases where acronym is outside, full form is inside bracket
#   Currently, excludes numbers
#
# ============================================================================



import re
from typing import Dict, List
import nltk

# utility function, dependancy for regex_parser
def isnumeric(arg: str) -> bool:
    """ Unlike string.isdigit, checks if x is coercible to float """
    try:
        arg = float(arg)
        return True
    except ValueError:
        return False


def regex_parser(text: str) -> Dict[str, List[str]]:
    """ """
    stopwords = set(nltk.corpus.stopwords.words('english'))
    sents = nltk.sent_tokenize(text)
    abbreviations = {}
    query_brackets = re.compile(r'\(([^\)]+)\)')
    for sent_ in sents:
        matches = list(re.finditer(query_brackets, sent_))
        if matches:
            for match_ in matches:
                
                match = (match_.group())[1:-1]

                sent = sent_[0: match_.start() - 1]
                
                # exclude numbers
                if isnumeric(match):
                    continue

                # check that the phrase is not single letter
                if len(match) <= 1:
                    continue

                # check that atleast one alphabetic char is capital form
                if not re.findall(r'[A-Z]', match):
                    continue

                alpha = re.sub(r'[^a-zA-Z]', '', match)
                pattern = [alpha[i:] for i in range(len(alpha))]
                pattern = ['|'.join(i) for i in pattern]
                pattern = [i[0] + r'(?:\w+|\b' + i[1:] + ')' for i in pattern]
                query_final = r'\b' + '?(?:.)'.join(pattern) # don't re.compile here
                putatives = re.findall(query_final, sent, re.IGNORECASE)
                
                if not putatives:
                    tokens = nltk.word_tokenize(sent)
                    tokens = tokens[max(0, len(match)):]
                    trim_sent = ' '.join([tok for tok in tokens
                                    if tok not in stopwords
                                    and not isnumeric(tok)])
                    pattern = [alpha[i:] for i in range(len(alpha))]
                    pattern = ['|'.join(i) for i in pattern]
                    pattern = [i[0] + r'(?:\w+|\b' + i[1:] + ')' for i in pattern]
                    query_final = r'\b' + '?(?:.*?)'.join(pattern)
                    putatives = re.findall(query_final, sent, re.IGNORECASE)
                        

                if not putatives:
                    print('{}: {}:  {}'.format(match, query_final, sent))

                abbreviations[match] = putatives
    return abbreviations


def hmm_parser(text: str):
    """ """
    pass
