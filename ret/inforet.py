from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import re
import math
from collections import defaultdict, Counter
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet, stopwords
from nltk import pos_tag
from difflib import get_close_matches
import nltk
import numpy as np

nltk.download('wordnet',quiet=True)
nltk.download('stopwords',quiet=True)
nltk.download('averaged_perceptron_tagger',quiet=True)
nltk.download('punkt',quiet=True)

app = Flask(__name__)

# Configure the exact path to your images
# IMAGE_FOLDER = r'C:\Users\afrat\OneDrive\Desktop\mos\MOS-2\crawler\image_crawler\static\images'
IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), 'static/images')
print(IMAGE_FOLDER)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

@app.route('/images/<path:filename>')
def serve_static(filename):
    """Serve files from your specific static directory"""
    return send_from_directory(IMAGE_FOLDER, filename)

class TextPreProcessor:
    def __init__(self, use_stemming=False, use_lemmatization=True):
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.stemmer = PorterStemmer() if use_stemming else None
        self.stopwords = set(stopwords.words('english'))

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        pos_tags = pos_tag(tokens)
        processed = []
        for word, tag in pos_tags:
            if word not in self.stopwords:
                if self.lemmatizer:
                    pos = self._get_wordnet_pos(tag)
                    word = self.lemmatizer.lemmatize(word, pos) if pos else word
                if self.stemmer:
                    word = self.stemmer.stem(word)
                processed.append(word)
        return processed

    def _get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        return None

class ImageSurrogateIndexer:
    def __init__(self):
        self.processor = TextPreProcessor()
        self.vocab = set()
        self.doc_freq = Counter()
        self.term_doc_matrix = defaultdict(lambda: defaultdict(int))
        self.docs = []
        self.doc_metadata = []
        self.term_to_index = {}
        self.index_to_term = {}
        self.seen_images = set()
        self.doc_lengths = []
        self.avg_doc_length = 0

    def parse_surrogates(self, file_path):
        current_doc = {}
        in_annotations = False
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith("Image:"):
                    image_path = line.split("Image: ")[1]
                    if image_path in self.seen_images:
                        current_doc = {}
                        continue
                    self.seen_images.add(image_path)
                    
                    if current_doc:
                        self._add_document_terms(current_doc)
                    current_doc = {
                        'id': image_path,
                        'text': '',
                        'alt_text': '',
                        'annotations': ''
                    }
                    in_annotations = False
                elif line.startswith("Alt Text:") and current_doc:
                    current_doc['alt_text'] = line.split("Alt Text: ")[1]
                    current_doc['text'] += current_doc['alt_text'] + " "
                elif line == "Image Caption:" and current_doc:
                    in_annotations = True
                elif in_annotations and current_doc:
                    current_doc['annotations'] = line.replace('"','').replace('"','').strip()
                    current_doc['text'] += line.strip() + " "
                    # if '"' in line:
                    #     label = line.split('"')[1]
                    # else:
                    #     label = line[2:].split(",")[0].strip()
                    # if label:
                    #     current_doc['annotations'].append(label)
                    #     current_doc['text'] += label + " "
        
        if current_doc and current_doc.get('id') not in self.seen_images:
            self._add_document_terms(current_doc)

    def _add_document_terms(self, doc):
        processed = self.processor.preprocess(doc['text'])
        self.docs.append(processed)
        self.doc_metadata.append(doc)
        self.doc_lengths.append(len(processed))
        
        unique_terms = set(processed)
        self.vocab.update(unique_terms)
        for term in unique_terms:
            self.doc_freq[term] += 1

    def build_index(self):
        self.term_to_index = {term: idx for idx, term in enumerate(sorted(self.vocab))}
        self.index_to_term = {idx: term for term, idx in self.term_to_index.items()}
        
        self.term_doc_matrix = defaultdict(lambda: defaultdict(int))
        for doc_id, doc in enumerate(self.docs):
            for term in doc:
                term_id = self.term_to_index[term]
                self.term_doc_matrix[term_id][doc_id] += 1
        
        self.avg_doc_length = np.mean(self.doc_lengths) if self.doc_lengths else 0

    def expand_query(self, query):
        """Expand query with synonyms, plurals, and similar terms"""
        terms = self.processor.preprocess(query)
        expanded = set(terms)
        
        for term in terms:
            if term.endswith('s'):
                expanded.add(term[:-1])
            else:
                expanded.add(term + 's')
            
            for syn in wordnet.synsets(term):
                for lemma in syn.lemmas():
                    expanded.add(lemma.name().replace('_', ' '))
            
            close_matches = get_close_matches(term, self.vocab, n=3, cutoff=0.6)
            expanded.update(close_matches)
        
        return list(expanded)

    def _search_vsm(self, query_terms):
        """Vector Space Model with TF-IDF and cosine similarity"""
        query_vector = np.zeros(len(self.vocab))
        term_to_index = self.term_to_index
        query_length = len(query_terms)
        
        if query_length == 0:
            return []
        
        # Calculate TF-IDF for query
        for term in query_terms:
            if term in term_to_index:
                tf = query_terms.count(term) / query_length
                df = self.doc_freq.get(term, 1)
                idf = math.log(len(self.docs) / (df + 1))
                query_vector[term_to_index[term]] = tf * idf
        
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector /= query_norm
        
        # Calculate cosine similarity
        scores = np.zeros(len(self.docs))
        for term_id, weight in enumerate(query_vector):
            if weight == 0:
                continue
            for doc_id, tf in self.term_doc_matrix[term_id].items():
                scores[doc_id] += weight * tf
        
        # Normalize scores by document length
        for doc_id in range(len(self.docs)):
            if self.doc_lengths[doc_id] > 0:
                scores[doc_id] /= self.doc_lengths[doc_id]
        
        return self._format_results(scores)

    def _search_bm25(self, query_terms, k1=2.5, b=0.8):
        """BM25 ranking algorithm"""
        scores = np.zeros(len(self.docs))
        term_to_index = self.term_to_index
        
        for term in query_terms:
            if term in term_to_index:
                term_id = term_to_index[term]
                df = self.doc_freq.get(term, 1)
                idf = math.log((len(self.docs) - df + 0.5) / (df + 0.5))
                
                for doc_id, tf in self.term_doc_matrix[term_id].items():
                    doc_length = self.doc_lengths[doc_id]
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_length / self.avg_doc_length))
                    scores[doc_id] += idf * (numerator / denominator)
        
        return self._format_results(scores)

    def _search_unigram(self, query_terms, alpha=0.1):
        """Unigram Language Model with Laplace smoothing"""
        scores = np.zeros(len(self.docs))
        vocab_size = len(self.vocab)
        
        for doc_id in range(len(self.docs)):
            doc_length = self.doc_lengths[doc_id]
            doc_score = 0.0
            
            for term in query_terms:
                if term in self.term_to_index:
                    term_id = self.term_to_index[term]
                    tf = self.term_doc_matrix[term_id].get(doc_id, 0)
                    # Laplace smoothing
                    doc_score += math.log((tf + 1) / (doc_length + vocab_size))
                else:
                    # Laplace smoothing for unseen terms
                    doc_score += math.log(1 / (doc_length + vocab_size))
            
            scores[doc_id] = doc_score
        
        return self._format_results(scores)

    def _format_results(self, scores):
        """Format the results for JSON response"""
        results = []
        for doc_id in np.argsort(scores)[::-1]:
            if scores[doc_id] > 0:
                results.append({
                    'image_path': self.doc_metadata[doc_id]['id'],
                    'alt_text': self.doc_metadata[doc_id]['alt_text'],
                    'annotations': self.doc_metadata[doc_id]['annotations'],
                    'score': float(scores[doc_id])
                })
        return results

    def search_all(self, query=None, model='vsm'):
        if query is None:
            return [{
                'image_path': meta['id'],
                'alt_text': meta['alt_text'],
                'annotations': meta['annotations'],
                'score': 0.0
            } for meta in self.doc_metadata]
        
        original_terms = self.processor.preprocess(query)
        if not original_terms:
            return []
        
        # Try exact match first
        if model == 'vsm':
            results = self._search_vsm(original_terms)
        elif model == 'bm25':
            results = self._search_bm25(original_terms)
        elif model == 'unigram':
            results = self._search_unigram(original_terms)
        else:
            results = self._search_vsm(original_terms)
        
        if results:
            return results
        
        # Fallback to expanded query if no results
        expanded_terms = self.expand_query(query)
        if model == 'vsm':
            results = self._search_vsm(expanded_terms)
        elif model == 'bm25':
            results = self._search_bm25(expanded_terms)
        elif model == 'unigram':
            results = self._search_unigram(expanded_terms)
        
        if not results:
            # Final fallback to visual concept matching
            for doc_id, meta in enumerate(self.doc_metadata):
                annotation_text = ' '.join(meta['annotations']).lower()
                alt_text = meta['alt_text'].lower()
                for term in original_terms:
                    if term in annotation_text or term in alt_text:
                        results.append({
                            'image_path': meta['id'],
                            'alt_text': meta['alt_text'],
                            'annotations': meta['annotations'],
                            'score': 0.3  # Lower confidence score
                        })
                        break
        
        return sorted(results, key=lambda x: x['score'], reverse=True)

# Initialize the indexer
indexer = ImageSurrogateIndexer()
try:
    surrogates_path = os.path.join(os.path.dirname(__file__), 'static/textual_surrogates.txt')
    indexer.parse_surrogates(surrogates_path)
    indexer.build_index()
    print(f"Indexed {len(indexer.docs)} unique images")
    print(f"Vocabulary size: {len(indexer.vocab)} terms")
except FileNotFoundError:
    print(f"Error: Could not find file at {surrogates_path}")
except Exception as e:
    print(f"An error occurred: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '').strip()
    model = request.args.get('model', 'vsm')  # Default to VSM
    
    if query.lower() == 'all':
        results = indexer.search_all()
    else:
        results = indexer.search_all(query, model)
    
    return jsonify({
        'query': query,
        'model': model,
        'results': results,
        'count': len(results)
    })

if __name__ == '__main__':
    app.run(debug=True)