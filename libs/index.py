import nltk

from nltk.corpus import stopwords
from nltk.stem.api import StemmerI
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from pyroaring import BitMap


nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


class PositionalIndex:
    def __init__(self, max_word_delta: int = 3):
        self.max_word_delta: int = max_word_delta

        self.word2doc = {}
        self.word_doc_poses = {}
        self._current_docid = 0

        self._stops: dict[str, set] = {}
        self._stemmer: dict[str, StemmerI] = {
            "english": PorterStemmer()
        }

    def add(self, text: str) -> int:
        doc_id = self._current_docid
        self._current_docid += 1

        words = self.preprocess_text(text)
        for position, word in enumerate(words):
            self.word2doc.setdefault(word, BitMap()).add(doc_id)
            self.word_doc_poses.setdefault((word, doc_id), BitMap()).add(position)

        return doc_id

    def query(self, phrase: str) -> BitMap:
        words = list(self.preprocess_text(phrase))
        if not words:
            return BitMap()

        docs_bitmap = self.word2doc[words[0]].copy()
        for word in words[1:]:
            if word not in self.word2doc:
                return BitMap()
            docs_bitmap &= self.word2doc[word]

        if not docs_bitmap:
            return BitMap()

        result = BitMap()
        for doc_id in docs_bitmap:
            word_poses = [
                sorted(self.word_doc_poses[(word, doc_id)]) for word in words
            ]
            if self.phrase_in_poses(word_poses, self.max_word_delta):
                result.add(doc_id)

        return result

    def preprocess_text(self, text, lang: str = "english") -> list[str]:
        if lang not in self._stops:
            self._stops[lang] = set(stopwords.words(lang))
        words = word_tokenize(text)
        stemmer = self._stemmer.get("english", PorterStemmer())
        return [
            stemmer.stem(word) for word in words if word.isalnum() and word.lower() not in self._stops[lang]
        ]

    @staticmethod
    def phrase_in_poses(word2poses: list[list[int]], max_word_delta: int) -> bool:
        pointers = [0] * len(word2poses)
        positions_lengths = [len(positions) for positions in word2poses]

        while pointers[0] < positions_lengths[0]:
            current_pos = word2poses[0][pointers[0]]
            match = True

            for i in range(1, len(word2poses)):
                while pointers[i] < positions_lengths[i] and word2poses[i][pointers[i]] < current_pos:
                    pointers[i] += 1
                if pointers[i] == positions_lengths[i]:
                    return False
                next_pos = word2poses[i][pointers[i]]
                if next_pos - current_pos > max_word_delta:
                    match = False
                    break
                current_pos = next_pos

            if match:
                return True
            pointers[0] += 1

        return False
