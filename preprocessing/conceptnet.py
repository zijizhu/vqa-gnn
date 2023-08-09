import json
import nltk
import spacy
import pickle
import networkx as nx
from tqdm import tqdm
from spacy.tokens import Doc

relation_raw2merged = {
    'atlocation': 'atlocation',
    'locatednear': 'atlocation',
    'capableof': 'capableof',
    'causes': 'causes',
    'causesdesire': 'causes',
    'motivatedbygoal': '*causes',
    'createdby': 'createdby',
    'desires': 'desires',
    'antonym': 'antonym',
    'distinctfrom': 'antonym',
    'hascontext': 'hascontext',
    'hasproperty': 'hasproperty',
    'hassubevent': 'hassubevent',
    'hasfirstsubevent': 'hassubevent',
    'haslastsubevent': 'hassubevent',
    'hasprerequisite': 'hassubevent',
    'entails': 'hassubevent',
    'mannerof': 'hassubevent',
    'isa': 'isa',
    'instanceof': 'isa',
    'definedas': 'isa',
    'madeof': 'madeof',
    'notcapableof': 'notcapableof',
    'notdesires': 'notdesires',
    'partof': 'partof',
    'hasa': '*partof',
    'relatedto': 'relatedto',
    'similarto': 'relatedto',
    'synonym': 'relatedto',
    'usedfor': 'usedfor',
    'receivesaction': 'receivesaction'
}

merged_relations_set = set(list(
    val[1:]
    if val.startswith('*')
    else val
    for val in relation_raw2merged.values()))


def delete_pos(s):
    if s.endswith('/n') or s.endswith('/a') or s.endswith('/v') or s.endswith('/r'):
        s = s[:-2]
    return s


def clean_field(s: str):
    return s.split('/')[-1].lower()


def extract_english(
        cnet_raw_path: str,
        cnet_eng_csv_path: str,
        cnet_eng_vocab_path: str):
    concept_set = set()
    nrows = sum(1 for _ in open(cnet_raw_path, 'r', encoding='utf-8'))

    print('Extracting conceptnet english')
    with open(cnet_raw_path, 'r', encoding='utf8') as infile, \
            open(cnet_eng_csv_path, 'w', encoding='utf8') as outfile:
        for line in tqdm(infile, total=nrows):
            [_, rel, head, tail, data] = line.strip().split('\t')
            if head.startswith('/c/en/') and tail.startswith('/c/en/'):
                rel = clean_field(rel)
                head = clean_field(delete_pos(head))
                tail = clean_field(delete_pos(tail))

                data_dict = json.loads(data)

                if not head.replace('_', '').replace('-', '').isalpha():
                    continue
                if not tail.replace('_', '').replace('-', '').isalpha():
                    continue
                if rel not in relation_raw2merged:
                    continue

                rel = relation_raw2merged[rel]
                if rel.startswith('*'):
                    head, tail, rel = tail, head, rel[1:]

                outfile.write(
                    ','.join([rel, head, tail, str(data_dict['weight'])]) + '\n')
                concept_set.update([head, tail])

    with open(cnet_eng_vocab_path, 'w') as outfile:
        for word in concept_set:
            outfile.write(word + '\n')

    print(f'Extracted conceptnet csv file saved to {cnet_eng_csv_path}')
    print(f'Extracted concept vocabulary saved to {cnet_eng_vocab_path}')
    print()


def construct_graph(
        cnet_eng_csv_path: str,
        cnet_eng_vocab_path: str,
        cnet_graph_path: str,
        prune: bool = False):
    print('Generating ConceptNet graph file')

    blacklist = set(['uk', 'us', 'take', 'make', 'object', 'person', 'people'])

    id2concept = []
    concept2id = {}
    with open(cnet_eng_vocab_path, 'r', encoding='utf8') as infile:
        id2concept = [w.strip() for w in infile]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = list(merged_relations_set)
    relation2id = {r: i for i, r in enumerate(id2relation)}

    graph = nx.MultiDiGraph()

    nrows = sum(1 for _ in open(cnet_eng_csv_path, 'r', encoding='utf-8'))
    with open(cnet_eng_csv_path, 'r', encoding='utf8') as fin:
        added_edge = set()

        for line in tqdm(fin, total=nrows):
            [rel, subj, obj, weight] = line.strip().split(',')
            rel_id, subj_id, obj_id, = relation2id[rel], concept2id[subj], concept2id[obj]

            if prune and (subj in blacklist or object in blacklist or id2relation[rel] == 'hascontext'):
                continue
            if subj == obj:
                continue

            if (subj_id, obj_id, rel_id) not in added_edge:
                graph.add_edge(subj_id, obj_id, rel_id=rel_id, weight=weight)
                added_edge.add((subj_id, obj_id, rel_id))
                graph.add_edge(obj_id, subj_id,
                               rel_id=rel_id + len(relation2id), weight=weight)
                added_edge.add((obj_id, subj_id, rel_id + len(relation2id)))

    with open(cnet_graph_path, 'wb') as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
    print(f'Conceptnet graph object saved to {cnet_graph_path}')
    print()


nltk.download('stopwords', quiet=True)
nltk_stopwords = nltk.corpus.stopwords.words('english')


def create_pattern(doc: Doc):
    pronoun_list = set(['my', 'you', 'it', 'its', 'your', 'i', 'he',
                       'she', 'his', 'her', 'they', 'them', 'their', 'our', 'we'])

    blacklist = set([
        '-PRON-', 'actually', 'likely', 'possibly', 'want',
        'make', 'my', 'someone', 'sometimes_people', 'sometimes', 'would', 'want_to',
        'one', 'something', 'sometimes', 'everybody', 'somebody', 'could', 'could_be'
    ])

    # Filtering concepts that:
    # 1. Longer than 5 words
    # 2. starts or ends with pronoun
    # 3. consists of stopworks only
    def overlength(x): return len(x) >= 5
    def startswith_pronoun(x): return x[0].text in pronoun_list
    def endswith_pronoun(x): return x[-1].text in pronoun_list

    def is_stopword(x):
        return x.text in nltk_stopwords or x.lemma_ in nltk_stopwords or x.lemma_ in blacklist

    if overlength(doc) \
            or startswith_pronoun(doc) \
            or endswith_pronoun(doc) \
            or all([is_stopword(token) for token in doc]):
        return None
    return tuple(token.lemma_ for token in doc)


def load_cpnet_vocab(cnet_eng_vocab_path):
    with open(cnet_eng_vocab_path, 'r', encoding='utf8') as fin:
        cnet_eng_vocab = [l.strip() for l in fin]
    cnet_eng_vocab = [c.replace('_', ' ') for c in cnet_eng_vocab]
    return cnet_eng_vocab


def create_matcher_patterns(cnet_eng_vocab_path: str, matcher_patterns_path: str,):
    cnet_vocab = load_cpnet_vocab(cnet_eng_vocab_path)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
    docs = nlp.pipe(cnet_vocab)
    all_patterns = {}

    print('Create matcher patterns from conceptnet')
    for doc in tqdm(docs, total=len(cnet_vocab)):
        pattern = create_pattern(doc)
        if pattern is None:
            continue
        all_patterns['_'.join(doc.text.split(' '))] = pattern

    print('Created ' + str(len(all_patterns)) + ' patterns')
    with open(matcher_patterns_path, 'w', encoding='utf8') as outfile:
        json.dump(all_patterns, outfile)

