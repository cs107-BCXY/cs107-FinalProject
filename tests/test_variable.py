import unittest
from cs107-FinalProject.src.variable import Variable
# TODO: fix this import


class TestVariable(unittest.TestCase):
    def test_construction(self):
        """Check that Variable is initialized correctly.
        """
        x = Variable(3)
        self.assertEqual(x.val, 3)
        self.assertEqual(x.der, 1)


if __name__ == "__main__":
    unittest.main()