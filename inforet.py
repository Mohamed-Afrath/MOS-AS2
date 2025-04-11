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

nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt", quiet=True)

app = Flask(__name__)

# Example test cases (modify based on your dataset)
TEST_QUERIES = {
    "Women": [
        "2023-11-13-12-34-18-576x720.jpg",
        "2023-11-21-11-41-34-576x720.jpg",
        "2023-11-07-07-58-51-576x720.jpg",
        "2025-02-25-05-43-39-576x720.jpg",
        "2024-02-02-07-47-33-576x864.jpg",
        "2023-10-27-08-13-03-576x720.jpg",
        "2025-03-08-11-51-02-576x576.jpeg",
        "2023-10-07-15-28-24-576x434.jpg",
        "2023-11-24-08-47-09-576x720.jpg",
        "2025-03-19-05-38-24-576x372.jpeg",
        "2023-08-16-11-30-41-576x323.jpg",
        "2024-01-24-10-56-28-576x384.jpg",
        "2023-10-26-09-10-33-576x720.jpg",
        "2024-06-14-06-53-08-576x720.jpg",
        "2023-08-16-11-31-49-576x323.jpg",
        "2025-03-11-09-40-32-576x1016.jpeg",
        "2024-02-01-09-25-09-576x720.jpg",
        "2023-11-10-09-08-11-576x720.jpg",
        "2024-01-31-08-34-04-576x720.jpg",
        "2023-10-26-09-12-18-1-576x720.jpg",
        "2023-11-07-07-59-13-576x720.jpg",
        "2024-04-13-10-18-57-576x720.jpg",
        "2024-06-08-13-24-18-576x864.jpg",
        "2024-04-19-06-35-53-576x720.jpg",
        "2023-10-25-09-15-00-576x384.jpg",
        "2024-01-31-08-33-05-576x720.jpg",
        "2023-10-26-09-10-54-576x720.jpg",
        "2023-11-07-08-00-55-576x720.jpg",
        "2023-10-31-08-02-49-576x720.jpg",
        "2023-11-10-09-12-52-576x720.jpg",
        "2025-03-19-05-40-41-576x576.png",
        "2024-07-02-11-30-12-576x434.png",
        "2023-12-28-10-16-55-576x720.jpg",
        "2024-04-23-07-26-34-576x720.jpg",
        "2024-02-16-10-29-02-576x720.jpg",
        "2024-01-16-05-51-06-576x720.jpg",
        "2024-03-11-08-25-39-576x864.jpg",
        "2024-04-09-07-03-07-576x864.jpg",
        "2023-10-31-07-55-58-576x720.jpg",
        "2023-07-22-19-13-32-576x576.jpg",
        "2024-06-08-09-14-01-576x720.jpg",
        "2024-02-09-08-07-27-576x720.jpg",
        "2023-10-24-09-33-44-576x720.jpg",
        "2024-02-16-10-29-18-576x720.jpg",
        "2024-06-08-09-12-30-576x720.jpg",
        "2024-12-21-10-21-41-576x720.jpg",
        "2023-11-13-12-35-44-576x720.jpg",
        "2024-10-26-08-48-17-576x720.jpg",
        "2024-06-06-06-57-07-576x720.jpg",
        "2024-01-19-09-00-55-576x384.jpg",
        "2023-12-28-10-20-08-576x720.jpg",
        "2024-04-24-07-33-15-576x720.jpg",
        "2024-04-24-07-33-02-576x384.jpg",
        "2024-04-19-06-36-10-576x720.jpg",
        "2025-03-15-05-31-24-576x576.jpeg",
        "2024-04-29-07-08-07-576x720.jpg",
        "2025-03-10-09-48-19-576x329.png",
        "2024-06-13-08-15-36-576x720.jpg",
        "2024-06-06-06-56-47-576x720.jpg",
        "2025-02-15-06-28-43-576x384.jpg",
        "2025-01-27-09-26-16-576x360.jpg",
        "2024-06-12-07-04-08-576x384.jpg",
        "2024-04-04-21-49-36-576x576.jpeg",
        "2024-06-05-08-00-39-576x720.jpg",
        "2023-10-31-07-52-43-576x745.jpg",
        "2023-11-10-09-16-59-576x720.jpg",
        "2024-06-08-09-13-15-576x720.jpg",
        "2024-09-30-09-09-37-576x720.jpg",
        "2024-02-07-11-58-05-576x384.jpg",
        "2023-08-09-13-55-36-576x864.jpg",
        "2024-08-31-05-44-22-576x720.jpg",
        "2024-12-16-07-40-10-576x384.jpg",
        "2024-04-13-10-19-42-576x720.jpg",
        "2024-04-09-07-02-34-576x720.jpg",
        "2024-04-19-06-35-34-576x720.jpg",
        "2024-06-08-09-15-32-576x384.jpg",
        "2025-02-18-09-41-13-576x329.png",
        "2024-03-07-01-37-48-576x432.jpg",
        "2024-03-07-08-03-20-576x720.jpg",
        "2023-11-14-08-36-56-576x720.jpg",
        "2024-06-13-08-17-25-576x818.jpg",
        "2022-06-30-11-15-16-576x720.jpg",
        "2024-06-08-09-15-11-576x384.jpg",
        "2024-08-12-09-53-56-576x720.jpg",
        "2025-02-25-05-43-08-576x720.jpg",
        "2024-04-24-07-31-49-576x720.jpg",
        "2025-01-28-08-03-34-576x720.jpg",
        "2024-06-13-08-18-02-576x720.jpg",
        "2025-02-06-11-09-52-576x765.jpg",
        "2024-06-11-07-17-38-576x720.jpg",
        "2024-12-05-08-59-10-576x720.jpg",
        "2024-04-29-07-09-14-576x720.jpg",
        "2024-04-29-07-08-29-576x720.jpg",
        "2024-01-16-05-46-58-576x384.jpg",
        "2024-06-14-06-52-21-576x720.jpg",
        "2024-06-12-07-01-28-576x720.jpg",
        "2024-04-19-06-34-47-576x720.jpg",
        "2024-04-15-06-37-22-576x720.jpg",
        "2024-12-16-07-44-30-576x720.jpg",
        "2024-06-08-09-14-24-576x720.jpg",
        "2024-10-26-08-48-32-576x720.jpg",
        "2024-04-24-07-32-15-576x384.jpg",
        "2024-06-07-07-35-27-576x720.jpg",
        "2024-06-12-07-04-25-576x720.jpg",
        "2023-11-06-07-32-47-576x720.jpg",
        "2024-06-07-07-34-04-576x720.jpg",
        "2024-06-07-07-32-13-576x720.jpg",
        "2024-06-06-06-54-26-576x720.jpg",
        "2024-06-14-06-52-06-576x384.jpg",
        "2025-02-15-06-30-36-576x720.jpg",
        "2024-12-19-09-31-32-1-576x864.jpg",
        "2024-12-19-09-27-51-576x864.jpg",
        "2024-04-20-04-53-30-576x720.jpg",
        "2023-10-26-09-12-42-576x720.jpg",
        "2024-06-05-07-59-55-576x720.jpg",
        "2024-07-02-11-36-13-576x434.png",
        "2023-11-21-11-44-13-576x720.jpg",
        "2023-11-06-07-35-17-576x384.jpg",
        "2023-11-09-07-47-05-576x720.jpg",
        "2024-04-10-07-46-44-576x720.jpg",
        "2023-11-21-11-46-08-576x720.jpg",
        "2023-11-09-07-43-35-576x720.jpg",
        "2023-06-19-15-27-31-576x960.png",
        "2024-11-11-10-19-39-576x720.jpg",
        "2024-04-20-04-50-29-576x720.jpg",
        "2023-11-10-09-26-49-576x720.jpg",
        "2023-12-06-08-05-50-576x720.jpg",
        "2023-07-22-19-12-47-576x384.jpg",
        "2023-11-03-10-58-50-576x720.jpg",
        "2023-11-02-08-27-58-576x720.jpg",
        "2024-08-12-09-53-41-576x720.jpg",
        "2024-01-08-11-50-00-576x384.jpg",
        "2024-09-12-09-13-10-576x720.jpg",
        "2024-01-22-09-13-10-576x384.jpg",
        "2023-10-25-09-12-43-576x720.jpg",
        "2023-06-24-17-48-09-576x576.jpg",
        "2023-09-25-15-33-09-576x864.jpg",
        "2023-09-25-15-33-39-576x384.jpg",
        "2023-11-06-07-32-09-576x720.jpg",
        "2023-11-09-07-51-40-576x720.jpg",
        "2023-06-03-10-44-01-576x323.jpg",
        "2023-08-21-13-15-42-576x864.jpg",
        "2023-06-19-07-21-35-576x1056.png",
        "2025-03-11-09-40-45-576x576.jpeg",
        "2023-11-07-07-58-26-576x720.jpg",
        "2023-10-26-09-15-14-576x720.jpg",
        "2023-11-09-07-43-50-576x720.jpg",
        "2023-10-23-09-00-17-576x720.jpg",
        "2024-04-29-07-07-53-576x720.jpg",
        "2023-11-25-09-46-20-576x720.jpg",
        "2023-11-28-07-54-42-576x720.jpg",
        "2024-01-24-22-40-03-576x434.jpg",
        "2023-11-24-08-41-06-576x720.jpg",
        "2023-05-17-06-24-18-576x384.jpg",
        "2025-03-08-11-50-21-576x329.jpeg",
        "2025-01-30-09-11-11-576x576.png",
        "2024-09-30-09-02-08-576x864.jpg",
        "2023-11-02-08-20-03-576x720.jpg",
        "2025-03-18-09-24-51-576x432.jpeg",
        "2023-11-06-07-33-01-576x720.jpg",
        "2023-04-05-11-51-25-576x864.jpg",
        "2023-06-23-11-16-17-576x432.jpg",
        "2023-11-09-07-41-08-576x720.jpg",
        "2024-03-08-08-04-51-576x720.jpg",
        "2023-10-25-09-15-19-576x720.jpg",
        "2024-06-05-07-58-22-576x720.jpg",
        "2023-11-06-07-37-31-576x720.jpg",
        "2023-06-29-20-57-03-576x323.jpg",
        "2024-07-02-11-30-41-576x434.png",
        "2023-06-29-10-06-23-576x768.jpg",
        "2023-11-24-08-40-14-576x720.jpg",
        "2024-01-05-10-15-40-576x818.jpg",
        "2024-02-03-10-02-26-576x864.jpg",
        "2024-02-03-10-02-08-576x768.jpg",
        "2025-03-12-10-04-54-576x432.jpeg",
        "2023-11-01-08-18-44-576x384.jpg",
        "2025-03-26-07-34-46-576x372.png",
        "2023-11-22-09-51-27-576x720.jpg",
        "2024-03-01-02-44-14-576x432.jpg",
        "2024-04-13-10-21-17-576x720.jpg",
        "2024-01-24-10-55-47-576x384.jpg",
        "2023-12-01-08-50-48-576x720.jpg",
        "2022-06-30-11-25-33-576x864.jpg",
        "2024-04-15-06-43-56-576x720.jpg",
        "2024-06-07-07-34-50-576x720.jpg",
        "2024-06-05-07-58-44-576x720.jpg",
        "2024-04-20-04-50-43-576x720.jpg",
        "2024-04-15-06-37-10-576x720.jpg",
        "2023-06-26-12-04-14-576x768.jpg",
        "2024-06-30-23-23-00-576x434.jpg",
        "2024-06-07-07-32-38-576x720.jpg",
        "2024-01-30-08-38-43-576x720.jpg",
        "2024-06-11-07-13-46-576x720.jpg",
        "2024-06-05-07-59-08-576x720.jpg",
        "2024-05-27-08-00-51-576x720.jpg",
        "2025-03-20-11-07-09-576x310.jpeg",
        "2024-06-11-07-16-17-576x384.jpg",
        "2024-07-02-00-46-51-576x432.jpg",
        "2024-12-16-07-40-52-576x384.jpg",
    ],
    "Black Dress": [
        "2025-02-25-05-43-08-576x720.jpg",
        "2024-11-11-10-19-39-576x720.jpg",
        "2024-02-03-10-02-26-576x864.jpg",
        "2025-01-28-08-03-34-576x720.jpg",
        "2024-09-30-09-09-37-576x720.jpg",
        "2024-12-16-07-44-30-576x720.jpg",
        "2024-10-26-08-48-32-576x720.jpg",
        "2024-06-06-06-54-15-576x720.jpg",
        "2024-06-08-09-15-11-576x384.jpg",
        "2024-12-05-08-59-10-576x720.jpg",
        "2024-11-11-10-19-07-576x720.jpg",
        "2024-12-16-07-44-13-576x720.jpg",
        "2024-08-12-09-53-41-576x720.jpg",
        "2023-11-13-12-35-44-576x720.jpg",
        "2024-06-10-10-39-49-576x384.jpg",
        "2024-02-07-11-58-05-576x384.jpg",
        "2024-08-31-05-44-22-576x720.jpg",
        "2024-01-31-08-33-05-576x720.jpg",
        "2024-04-19-06-34-47-576x720.jpg",
        "2025-02-25-05-43-39-576x720.jpg",
        "2024-06-12-07-04-25-576x720.jpg",
        "2024-01-24-10-56-28-576x384.jpg",
        "2024-06-11-07-16-17-576x384.jpg",
        "2023-06-22-14-03-51-576x749.jpg",
        "2025-02-15-06-30-36-576x720.jpg",
        "2024-10-26-08-48-17-576x720.jpg",
        "2024-12-21-10-21-41-576x720.jpg",
        "2023-11-09-07-51-40-576x720.jpg",
        "2023-11-24-08-41-06-576x720.jpg",
        "2023-11-24-08-40-14-576x720.jpg",
        "2024-02-09-08-07-27-576x720.jpg",
        "2023-12-28-10-20-08-576x720.jpg",
        "2024-04-15-06-37-10-576x720.jpg",
        "2024-04-24-07-31-49-576x720.jpg",
        "2024-06-07-07-35-27-576x720.jpg",
        "2023-07-01-19-14-09-576x323.jpg",
        "2024-06-06-06-54-26-576x720.jpg",
        "2024-06-07-07-32-13-576x720.jpg",
        "2024-06-14-06-52-06-576x384.jpg",
        "2024-01-17-08-19-57-576x720.jpg",
        "2024-06-08-09-14-48-576x720.jpg",
        "2025-02-11-05-25-59-576x360.jpg",
        "2024-05-24-07-14-49-576x720.jpg",
        "2024-06-06-06-55-07-576x720.jpg",
        "2024-12-24-09-07-10-576x384.jpg",
        "2024-04-20-04-50-29-576x720.jpg",
        "2025-02-26-08-13-46-576x384.jpg",
        "2023-08-01-09-17-06-576x384.jpg",
        "2023-10-22-18-24-14-576x434.jpg",
        "2023-06-13-09-01-02-576x384.jpg",
        "2023-07-01-19-14-19-576x323.jpg",
        "2025-02-19-05-17-12-576x384.jpg",
        "2025-02-11-05-23-44-576x763.jpg",
        "2023-11-21-11-46-08-576x720.jpg",
        "2023-11-06-07-35-17-576x384.jpg",
        "2025-02-04-09-29-54-576x360.jpg",
        "2023-06-09-21-16-57-576x323.jpg",
        "2023-08-05-20-14-13-576x1028.jpg",
        "2023-06-08-21-46-17-576x323.jpg",
        "2023-07-27-11-55-53-576x864.jpg",
        "2023-11-09-07-43-50-576x720.jpg",
        "2023-10-26-09-10-33-576x720.jpg",
        "2023-12-06-08-05-50-576x720.jpg",
        "2023-08-16-11-30-41-576x323.jpg",
        "2025-01-16-06-05-12-576x360.jpg",
        "2023-10-22-18-24-36-576x434.jpg",
        "2023-06-09-21-17-03-576x323.jpg",
        "2023-12-27-08-43-33-576x384.jpg",
        "2023-07-16-12-50-43-576x323.jpg",
        "2025-01-22-08-31-46-576x384.jpg",
        "2024-02-16-10-29-02-576x720.jpg",
        "2024-08-02-14-16-14-576x323.jpg",
        "2024-06-27-11-25-02-576x864.png",
        "2023-12-11-08-35-56-576x720.jpg",
        "2024-12-26-09-24-00-576x433.jpg",
        "2025-02-15-06-29-04-576x384.jpg",
        "2025-03-06-08-05-19-576x384.jpg",
        "2024-12-10-10-04-06-576x384.jpg",
        "2025-03-12-10-04-54-576x432.jpeg",
        "2023-11-06-07-33-01-576x720.jpg",
        "2023-11-01-08-18-44-576x384.jpg",
        "2024-03-08-08-04-51-576x720.jpg",
        "2024-08-12-09-53-56-576x720.jpg",
        "2024-12-26-09-24-17-576x384.jpg",
        "2022-09-23-08-31-24-576x384.jpg",
        "2023-11-10-09-16-59-576x720.jpg",
        "2024-06-12-07-04-08-576x384.jpg",
        "2024-04-13-10-21-17-576x720.jpg",
        "2023-11-22-09-51-27-576x720.jpg",
        "2024-06-05-08-00-39-576x720.jpg",
        "2023-06-09-21-16-00-576x323.jpg",
        "2025-03-01-05-51-48-576x434.jpg",
        "2025-03-05-07-15-02-576x384.jpg",
        "2025-02-06-05-23-23-576x384.jpg",
        "2024-01-29-12-08-47-576x384.jpg",
        "2024-05-24-07-11-22-576x720.jpg",
        "2024-10-18-15-24-38-576x765.jpg",
        "2024-01-16-05-46-58-576x384.jpg",
        "2024-06-11-07-17-38-576x720.jpg",
        "2024-06-12-07-01-28-576x720.jpg",
        "2024-06-05-07-57-41-576x720.jpg",
        "2024-04-13-10-18-21-576x720.jpg",
        "2023-06-08-21-46-33-576x323.jpg",
        "2024-08-20-08-01-12-576x384.jpg",
        "2024-04-24-07-32-15-576x384.jpg",
        "2023-04-28-08-24-04-576x384.jpg",
        "2024-01-05-10-16-48-576x864.jpg",
        "2025-03-14-05-59-33-576x768.jpeg",
        "2024-03-31-01-16-44-576x1024.jpg",
        "2023-07-01-14-10-52-576x323.jpg",
        "2024-01-17-08-19-31-576x720.jpg",
        "2023-11-09-07-43-35-576x720.jpg",
        "2023-06-08-21-46-39-576x323.jpg",
        "2024-04-20-04-53-30-576x720.jpg",
        "2023-06-19-07-21-35-576x1056.png",
        "2023-08-15-02-21-30-576x873.jpg",
        "2023-08-23-11-43-12-576x1028.jpg",
        "2023-06-13-17-10-38-576x323.jpg",
        "2023-11-10-09-26-49-576x720.jpg",
        "pixnio.gif",
        "2023-09-18-08-06-07-576x384.jpg",
        "2024-10-03-11-37-41-576x384.jpg",
        "2024-01-27-00-18-44-576x323.jpg",
        "2023-10-26-09-12-18-1-576x720.jpg",
        "2024-02-27-08-06-59-576x384.jpg",
        "2024-09-30-09-05-47-576x384.jpg",
        "2024-09-12-09-13-10-576x720.jpg",
        "2025-02-21-05-35-03-576x384.jpg",
        "2025-03-25-05-09-53-576x768.jpeg",
        "2024-05-01-13-43-48-576x384.jpg",
        "2023-07-18-19-55-35-576x323.jpg",
        "2025-02-20-05-36-32-576x384.jpg",
        "2024-01-31-16-13-17-576x434.jpg",
        "2025-03-06-08-10-27-576x384.jpg",
        "2024-03-01-02-30-00-576x432.jpg",
        "2024-08-26-07-59-52-576x864.jpg",
        "2023-07-23-13-38-19-576x877.jpg",
        "2024-04-19-06-35-53-576x720.jpg",
        "2024-11-07-09-15-55-576x384.jpg",
        "2023-11-27-09-18-11-576x384.jpg",
        "2023-06-26-20-33-00-576x323.jpg",
        "2023-09-20-07-06-31-576x864.jpg",
        "2023-08-26-20-17-38-576x368.jpg",
        "2023-06-02-10-29-38-576x384.jpg",
        "2024-07-27-05-48-21-576x384.jpg",
        "2024-12-23-13-26-45-576x435.png",
        "2024-05-25-07-08-39-576x384.jpg",
        "2024-05-28-20-45-14-576x384.jpg",
        "2024-01-05-10-15-40-576x818.jpg",
        "2023-03-05-22-30-23-576x864.jpg",
        "2023-08-05-17-40-54-576x768.jpg",
        "2024-02-08-16-06-19-576x384.jpg",
        "2023-06-23-11-16-17-576x432.jpg",
        "2024-03-08-08-09-10-576x384.jpg",
        "2024-02-16-10-29-18-576x720.jpg",
        "2025-02-20-05-35-22-576x384.jpg",
        "2025-02-19-05-16-57-576x384.jpg",
        "2024-12-06-09-17-52-576x384.jpg",
        "2024-08-29-05-07-13-576x864.jpg",
        "2024-01-24-22-40-03-576x434.jpg",
        "2024-12-18-09-53-53-576x864.jpg",
        "2024-04-24-07-33-15-576x720.jpg",
        "2024-06-13-08-18-02-576x720.jpg",
        "2024-04-29-07-08-29-576x720.jpg",
        "2025-01-07-09-10-01-576x384.jpg",
        "2025-02-06-11-09-52-576x765.jpg",
        "2025-02-13-09-02-19-576x434.jpg",
        "2024-06-14-06-52-21-576x720.jpg",
        "2025-02-21-05-39-58-576x384.jpg",
        "2022-09-23-08-32-25-576x384.jpg",
        "2022-11-29-11-32-54-576x384.jpg",
        "2022-07-22-09-22-52-576x720.jpg",
        "2024-03-09-11-04-07-576x720.jpg",
        "2022-08-12-07-49-07-576x864.jpg",
        "2024-09-16-23-15-09-576x432.jpg",
        "2025-01-28-09-12-32-576x576.jpg",
        "2024-04-10-07-46-44-576x720.jpg",
        "2024-10-25-08-57-18-576x864.jpg",
        "2025-02-01-01-18-49-576x434.jpg",
        "2024-08-14-04-46-57-576x864.jpg",
    ],
    "Sunset": [
        "2024-10-23-08-28-34-576x384.jpg",
        "2023-10-31-19-35-36-576x864.jpg",
        "2024-01-10-12-51-16-576x434.jpg",
        "2024-01-27-09-23-33-576x384.jpg",
        "2024-12-31-19-07-50-576x323.jpg",
        "2023-09-08-07-04-55-576x834.jpg",
        "2024-01-07-11-05-33-576x384.jpg",
        "2024-03-12-01-38-04-576x432.jpg",
        "2023-12-17-11-08-46-576x720.jpg",
        "2025-02-18-09-50-27-576x384.jpg",
        "2024-11-04-16-09-14-576x323.jpg",
        "2023-10-24-09-32-03-576x864.jpg",
        "2023-07-12-16-59-37-576x576.jpg",
        "2024-02-12-11-59-44-576x432.jpg",
        "2022-05-18-08-48-58-576x745.jpg",
        "2022-07-22-09-20-28-576x720.jpg",
        "2024-03-01-02-43-01-576x445.jpg",
        "2022-09-15-08-49-30-576x720.jpg",
        "2024-12-06-09-17-18-576x384.jpg",
        "2024-10-16-07-57-51-576x384.jpg",
        "2024-09-07-09-09-50-576x384.jpg",
        "2024-08-27-00-29-54-576x1024.jpg",
        "2022-10-29-10-50-18-576x720.jpg",
        "2022-05-28-09-40-28-576x720.jpg",
        "2024-02-12-11-51-26-576x432.jpg",
        "2024-10-22-08-17-02-576x384.jpg",
        "2022-04-23-09-00-18-576x720.jpg",
        "2022-04-22-05-51-30-576x720.jpg",
        "2022-06-03-07-46-35-576x384.jpg",
        "2022-10-22-09-18-50-576x720.jpg",
        "2024-01-10-12-51-41-576x576.jpg",
        "2023-07-05-08-13-33-576x864.jpg",
        "2023-10-25-09-11-50-576x864.jpg",
        "2024-07-26-22-44-52-576x1024.jpg",
        "2023-10-27-08-06-24-576x864.jpg",
        "2023-09-20-10-18-06-576x384.jpg",
        "2023-08-20-22-01-32-576x323.jpg",
        "2023-09-19-15-37-54-576x323.jpg",
        "2023-07-10-07-52-48-576x384.jpg",
        "2025-02-26-08-17-07-576x384.jpg",
        "2024-05-30-11-00-08-576x323.jpg",
        "2024-08-09-11-41-21-576x324.jpg",
        "2023-08-06-14-42-26-576x323.jpg",
        "2024-12-18-09-57-56-576x384.jpg",
        "2023-09-03-15-03-51-576x384.jpg",
    ],
}

# Configure the exact path to your images
# IMAGE_FOLDER = r'C:\Users\afrat\OneDrive\Desktop\mos\MOS-2\crawler\image_crawler\static\images'
IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), "static/images")
os.makedirs(IMAGE_FOLDER, exist_ok=True)


@app.route("/images/<path:filename>")
def serve_static(filename):
    """Serve files from your specific static directory"""
    return send_from_directory(IMAGE_FOLDER, filename)


class TextPreProcessor:
    def __init__(self, use_stemming=False, use_lemmatization=True):
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.stemmer = PorterStemmer() if use_stemming else None
        self.stopwords = set(stopwords.words("english"))

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
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
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
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

        with open(file_path, "r", encoding="utf-8") as f:
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
                        "id": image_path,
                        "text": "",
                        "alt_text": "",
                        "annotations": "",
                    }
                    in_annotations = False
                elif line.startswith("Alt Text:") and current_doc:
                    current_doc["alt_text"] = line.split("Alt Text: ")[1]
                    current_doc["text"] += current_doc["alt_text"] + " "
                elif line == "Image Caption:" and current_doc:
                    in_annotations = True
                elif in_annotations and current_doc:
                    current_doc["annotations"] = (
                        line.replace('"', "").replace('"', "").strip()
                    )
                    current_doc["text"] += line.strip() + " "
                    # if '"' in line:
                    #     label = line.split('"')[1]
                    # else:
                    #     label = line[2:].split(",")[0].strip()
                    # if label:
                    #     current_doc['annotations'].append(label)
                    #     current_doc['text'] += label + " "

        if current_doc and current_doc.get("id") not in self.seen_images:
            self._add_document_terms(current_doc)

    def _add_document_terms(self, doc):
        processed = self.processor.preprocess(doc["text"])
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
            if term.endswith("s"):
                expanded.add(term[:-1])
            else:
                expanded.add(term + "s")

            for syn in wordnet.synsets(term):
                for lemma in syn.lemmas():
                    expanded.add(lemma.name().replace("_", " "))

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
                    denominator = tf + k1 * (
                        1 - b + b * (doc_length / self.avg_doc_length)
                    )
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
                results.append(
                    {
                        "image_path": self.doc_metadata[doc_id]["id"],
                        "alt_text": self.doc_metadata[doc_id]["alt_text"],
                        "annotations": self.doc_metadata[doc_id]["annotations"],
                        "score": float(scores[doc_id]),
                    }
                )
        return results

    def evaluate_query(self, query, relevant_docs, model="vsm"):
        results = self.search_all(query, model)
        retrieved_docs = {os.path.basename(res["image_path"]) for res in results}
        relevant_docs = set(relevant_docs)

        # Calculate metrics
        tp = len(retrieved_docs & relevant_docs)
        fp = len(retrieved_docs - relevant_docs)
        fn = len(relevant_docs - retrieved_docs)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Calculate Average Precision (AP)
        ap = 0
        relevant_count = 0
        for i, res in enumerate(results):
            if res["image_path"] in relevant_docs:
                relevant_count += 1
                ap += relevant_count / (i + 1)
        ap /= len(relevant_docs) if len(relevant_docs) > 0 else 1

        return {
            "query": query,
            "model": model,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap": ap,
            "retrieved": len(retrieved_docs),
            "relevant": len(relevant_docs),
        }

    def search_all(self, query=None, model="vsm"):
        if query is None:
            return [
                {
                    "image_path": meta["id"],
                    "alt_text": meta["alt_text"],
                    "annotations": meta["annotations"],
                    "score": 0.0,
                }
                for meta in self.doc_metadata
            ]

        original_terms = self.processor.preprocess(query)
        if not original_terms:
            return []

        # Try exact match first
        if model == "vsm":
            results = self._search_vsm(original_terms)
        elif model == "bm25":
            results = self._search_bm25(original_terms)
        elif model == "unigram":
            results = self._search_unigram(original_terms)
        else:
            results = self._search_vsm(original_terms)

        if results:
            return results

        # Fallback to expanded query if no results
        expanded_terms = self.expand_query(query)
        if model == "vsm":
            results = self._search_vsm(expanded_terms)
        elif model == "bm25":
            results = self._search_bm25(expanded_terms)
        elif model == "unigram":
            results = self._search_unigram(expanded_terms)

        if not results:
            # Final fallback to visual concept matching
            for doc_id, meta in enumerate(self.doc_metadata):
                annotation_text = " ".join(meta["annotations"]).lower()
                alt_text = meta["alt_text"].lower()
                for term in original_terms:
                    if term in annotation_text or term in alt_text:
                        results.append(
                            {
                                "image_path": meta["id"],
                                "alt_text": meta["alt_text"],
                                "annotations": meta["annotations"],
                                "score": 0.3,  # Lower confidence score
                            }
                        )
                        break
        return sorted(results, key=lambda x: x["score"], reverse=True)


# Initialize the indexer
indexer = ImageSurrogateIndexer()
try:
    surrogates_path = os.path.join(
        os.path.dirname(__file__), "static/textual_surrogates.txt"
    )
    indexer.parse_surrogates(surrogates_path)
    indexer.build_index()
    print(f"Indexed {len(indexer.docs)} unique images")
    print(f"Vocabulary size: {len(indexer.vocab)} terms")
except FileNotFoundError:
    print(f"Error: Could not find file at {surrogates_path}")
except Exception as e:
    print(f"An error occurred: {str(e)}")


@app.route("/")
def home():
    return render_template("index.html")


# @app.route("/evaluate", methods=["GET"])
# def evaluate():
#     model = request.args.get("model", "vsm")  # Default to VSM
#     evaluation_results = []

#     for query, relevant_docs in TEST_QUERIES.items():
#         metrics = indexer.evaluate_query(query, relevant_docs, model)
#         evaluation_results.append(metrics)

#     return jsonify(
#         {
#             "model": model,
#             "results": evaluation_results,
#             "average_precision": sum(m["ap"] for m in evaluation_results)
#             / len(evaluation_results),
#         }
#     )


@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()
    model = request.args.get("model", "vsm")  # Default to VSM

    if query.lower() == "all":
        results = indexer.search_all()
    else:
        results = indexer.search_all(query, model)

    return jsonify(
        {"query": query, "model": model, "results": results, "count": len(results)}
    )


if __name__ == "__main__":
    app.run(debug=True)
