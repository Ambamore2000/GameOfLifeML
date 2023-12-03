"""
Unit tests for the patterns module.

This test suite verifies the correctness of the predefined patterns and
the functionality of the random pattern generator in the patterns module.
Each predefined pattern is tested for its specific structure, and the random
pattern generator is tested for generating patterns of the correct size and
content.
"""

import unittest
import numpy as np
from gameoflifeml.generate.patterns import GLIDER, BLINKER, TOAD, BEACON, generate_random_pattern

class TestPatterns(unittest.TestCase):
    """
    Test suite for verifying the patterns module.

    This suite contains tests for predefined patterns such as GLIDER, BLINKER, TOAD, and BEACON
    to ensure they are defined correctly. It also tests the functionality of the 
    generate_random_pattern function to ensure it generates patterns of the correct size and composition.
    """

    def test_predefined_patterns(self):
        """
        Test if predefined patterns are correctly defined.

        This test verifies that each predefined pattern (GLIDER, BLINKER, TOAD, and BEACON)
        matches its expected array structure.
        """
        self.assertTrue(np.array_equal(GLIDER, np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]])))
        self.assertTrue(np.array_equal(BLINKER, np.array([[1, 1, 1]])))
        self.assertTrue(np.array_equal(TOAD, np.array([[0, 1, 1, 1], [1, 1, 1, 0]])))
        self.assertTrue(np.array_equal(BEACON, np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]])))

    def test_random_pattern_size(self):
        """
        Test if the random pattern generator creates patterns of the correct size.

        This test checks whether the generate_random_pattern function returns a pattern
        that matches the specified size.
        """
        size = (10, 15)
        pattern = generate_random_pattern(size)
        self.assertEqual(pattern.shape, size)

    def test_random_pattern_content(self):
        """
        Test if the random pattern contains only 0s and 1s.

        This test ensures that the generate_random_pattern function only includes 0s and 1s
        in the generated pattern.
        """
        size = (5, 5)
        pattern = generate_random_pattern(size)
        unique_elements = np.unique(pattern)
        self.assertTrue(np.all(np.isin(unique_elements, [0, 1])))

if __name__ == '__main__':
    unittest.main()
