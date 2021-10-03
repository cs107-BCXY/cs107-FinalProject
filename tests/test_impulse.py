# This file is for checking TravisCI and CodeCov, and will be removed later.
import unittest

from src.code import step_impulse


class TestStepImpulse(unittest.TestCase):

    def test_positive(self):
        self.assertEqual(step_impulse(3), 1)


if __name__ == '__main__':
    unittest.main()
