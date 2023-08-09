import os
from pathlib import Path

root_dir = Path(os.path.dirname(__file__))
data_dir = root_dir /  'data'

conceptnet_raw_path = data_dir / 'conceptnet-assertions-5.7.0.csv'
conceptnet_eng_csv_path = data_dir / 'conceptnet_english_cleaned.csv'
conceptnet_eng_vocab_path = data_dir / 'conceptnet_english_vocab.txt'
conceptnet_eng_graph_path = data_dir / 'conceptnet_english_graph.pickle'
conceptnet_eng_matcher_patterns_path = data_dir / 'conceptnet_english_matcher_patterns.txt'
