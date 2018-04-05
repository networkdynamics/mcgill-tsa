import wordninja
import preprocessor
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from autocorrect import spell
from string import punctuation
from progress.bar import ShadyBar

# globals
LEMMATIZER = WordNetLemmatizer()
TOKENIZER = TweetTokenizer(preserve_case=False, reduce_len=True)
PUNCTUATION = set(punctuation)

# helper
def should_be_spell_corrected(string):
    return len(set(string).intersection(PUNCTUATION)) == 0

#Preprocessing functions in the pipeline order.
def tokenize(text):
    return TOKENIZER.tokenize(text)

def remove_mentions(tokens):
    return [t for t in tokens if not t.startswith('@')]

def filter_tokens(tokens, filter_list=set()):
    return [t for t in tokens if t not in filter_list]

def split_hashtags(tokens):
    """
    Applies hashtag splitting on list of tokens into most likely words.
    E.g., '#trumpisbad' -> ['#', 'trump', 'is', 'bad']
    :param tokens: list of strings
    :return: list of strings
    """
    new_toks = []
    for token in tokens:
        if token.startswith('#'):
            splits = wordninja.split(token[1:])
            new_toks.append('#')
            for w in splits:
                new_toks.append(w)
        else:
            new_toks.append(token)
    return new_toks

def autocorrect(tokens, skipwords=set()):
    """
    Applies autocorrect on list of strings (only if they don't have punctuation
    in them. E.g., 'dancin' -> 'dancing'.
    :param tokens: list of strings
    :param skipwords: set of words NOT to spellcheck.
    :return: list of strings
    """
    corrected = []

    # labelled looks like this: [steve, is, :), happy] -> [steve, is, $EMOTICON$, happy]
    labelled = map(preprocessor.tokenize, tokens)
    for token, label in zip(tokens, labelled):
        if should_be_spell_corrected(label) and not label in skipwords:
            corrected.append(spell(token))
        else:
            corrected.append(token)
    return corrected

def lemmatize(tokens):
    return list(map(LEMMATIZER.lemmatize, tokens))

# Primary application functions.
def preprocess_tweets(tweets, verbose=False, progress=True):
    # Pipeline is a list of tuples of functions and optional arguments.
    pipeline = [
        (tokenize, {}),
        (filter_tokens, {'filter_list': {'<br/>', '<br>'}}),
        (remove_mentions, {}),
        (split_hashtags, {}),
        (autocorrect, {'skipwords': {'lol', 'xbox'}}),
        (lemmatize, {}),
    ]
    bar = ShadyBar('Preprocessing MTSA', max=len(tweets))
    for tweet in tweets:
        uncorrected = apply_pipeline(pipeline[:-2], tweet.orig_text, verbose=verbose)
        tweet.uncorrected_tokens = uncorrected
        corrected = apply_pipeline(pipeline[-2:], uncorrected, verbose=verbose)
        tweet.corrected_tokens = corrected
        bar.next()
    bar.finish()

def apply_pipeline(pipeline, data, verbose=True):
    output = data
    if verbose:
        print(f'Current tweet text: {output}')

    for fun, args in pipeline:
        orig = output
        output = fun(output, **args)

        if verbose:
            print(f'Applying function \"{fun.__name__}\" with args {args} onto:')
            if type(orig) is not str:
                print('\tOrigin: ' + ', '.join(orig))
            print(f'\tResult: {", ".join(output)}')

    if verbose:
        print(f'FINAL OUTPUT: {output}\n\n')
    return output
