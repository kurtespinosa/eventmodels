import unittest

from src.model.SBNN import SBNN
class MyTestCase(unittest.TestCase):
    def test__get_enclosing_word_ind(self):
        sentencewords = ['Experimental', 'Design', ':', 'Human', 'umbilical', 'vein', 'and', 'dermal', 'microvascular', 'endothelial', 'cells', 'were', 'infected', 'with', 'replication-deficient', 'adenoviruses', 'encoding', 'survivin', '(', 'pAd-Survivin', ')', ',', 'green', 'fluorescent', 'protein', '(', 'pAd-GFP', ')', ',', 'or', 'a', 'phosphorylation-defective', 'survivin', 'Thr(34)', '-->', 'Ala', '(', 'pAd-T34A', ')', 'dominant', 'negative', 'mutant', '.']
        triggerword = "Thr(34)-->Ala"
        x = SBNN()
        result = x.get_enclosing_word_ind(sentencewords, triggerword)
        self.assertEqual(result, [33,34,35])



if __name__ == '__main__':
    unittest.main()
