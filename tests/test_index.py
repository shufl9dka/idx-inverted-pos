import unittest

from pyroaring import BitMap
from libs.index import PositionalIndex


class TestPositionalIndex(unittest.TestCase):
    def test_simple_phrase(self):
        index = PositionalIndex(max_word_delta=3)
        doc_id = index.add("I am happy buzzing bug")

        found = index.query("happy bug")
        self.assertEqual(found, BitMap({doc_id}))

        found = index.query("I am bug")
        self.assertEqual(found, BitMap({doc_id}))

        found = index.query("buzzing bee")
        self.assertEqual(found, BitMap())

    def test_multiple_docs(self):
        index = PositionalIndex(max_word_delta=3)

        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Lazy dogs sleep quickly in the brown fog",
            "A quick movement of the enemy will jeopardize six gunboat",
            "All the quick, brown animals jump over lazy dogs frequently",
        ]
        doc_ids = [index.add(text) for text in texts]

        self.assertEqual(
            index.query("lazy dog"),
            BitMap({doc_ids[0], doc_ids[1], doc_ids[3]})
        )

        self.assertEqual(
            index.query("brown fox"),
            BitMap({doc_ids[0]})
        )

        self.assertEqual(
            index.query("quick brown"),
            BitMap({doc_ids[0], doc_ids[3]})
        )


if __name__ == "__main__":
    unittest.main()
