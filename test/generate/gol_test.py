"""
Unit tests for the Game of Life module.

This test suite verifies the correctness of the functions in the Game of Life module.
It includes tests for generating the next generation, placing patterns, applying random transformations,
and generating data with specific patterns.
"""

import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
from gameoflifeml.generate.gol import next_generation, place_pattern, random_rotation, random_flip, generate_data
from gameoflifeml.generate.patterns import GLIDER

class TestGameOfLife(unittest.TestCase):
    """
    Unit tests for the Game of Life functions.

    This test suite verifies the correctness of the functions in the Game of Life module.
    It includes tests for generating the next generation, placing patterns, applying random transformations,
    and generating data with specific patterns.
    """

    def test_next_generation_glider(self):
        """
        Test the next generation of a glider pattern.

        This test verifies that the next generation of a glider pattern matches the expected state.
        """
        initial_state = np.array([  [0,0,0,0,0],
                                    [0,0,1,0,0],
                                    [0,0,0,1,0],
                                    [0,1,1,1,0],
                                    [0,0,0,0,0] ])
        
        expected_next_state = np.array([    [0,0,0,0,0],
                                            [0,0,0,0,0],
                                            [0,1,0,1,0],
                                            [0,0,1,1,0],
                                            [0,0,1,0,0] ])
        np.testing.assert_array_equal(next_generation(initial_state), expected_next_state)

    def test_place_pattern(self):
        """
        Test placing a pattern on a grid.

        This test checks if placing a pattern on a grid at a specified position results in the expected grid state.
        """
        grid = np.zeros((5, 5), dtype=int)
        pattern = GLIDER
        position = (2, 2)
        expected_grid = np.array([[0,0,0,0,0], 
                                  [0,0,0,0,0], 
                                  [0,0,0,1,0], 
                                  [0,0,0,0,1], 
                                  [0,0,1,1,1]])
        np.testing.assert_array_equal(place_pattern(grid, pattern, position), expected_grid)

    def test_random_rotation(self):
        """
        Test random rotation of a pattern.

        This test checks if randomly rotating a pattern produces one of the expected rotated states.
        """
        pattern = GLIDER
        expected_rotations = [
            GLIDER,  # Original
            np.rot90(GLIDER),  # Rotated 90 degrees
            np.rot90(GLIDER, 2),  # Rotated 180 degrees
            np.rot90(GLIDER, 3)  # Rotated 270 degrees
        ]

        for _ in range(100):
            rotated_pattern = random_rotation(pattern)
            self.assertTrue(any(np.array_equal(rotated_pattern, exp) for exp in expected_rotations))

    def test_random_flip(self):
        """
        Test random flipping of a pattern.

        This test checks if randomly flipping a pattern produces one of the expected flipped states.
        """
        pattern = GLIDER
        expected_flips = [
            GLIDER,  # Original
            np.fliplr(GLIDER),  # Flipped left-right
            np.flipud(GLIDER),  # Flipped up-down
            np.flipud(np.fliplr(GLIDER))  # Flipped both ways
        ]

        for _ in range(10):
            flipped_pattern = random_flip(pattern)
            self.assertTrue(any(np.array_equal(flipped_pattern, exp) for exp in expected_flips))

    def test_centered_pattern(self):
        """
        Test generating centered patterns.

        This test checks if generating centered patterns results in the expected initial generation.
        """
        size = 10
        steps = 5
        num_samples = 1
        data = generate_data(size, steps, num_samples, GLIDER, is_centered=True, is_transform=False)

        x_offset = (size - GLIDER.shape[0]) // 2
        y_offset = (size - GLIDER.shape[1]) // 2
        expected_first_gen = np.zeros((size, size), dtype=int)
        expected_first_gen[x_offset:x_offset+GLIDER.shape[0], y_offset:y_offset+GLIDER.shape[1]] = GLIDER

        np.testing.assert_array_equal(data[0], expected_first_gen)

    def test_non_centered_pattern(self):
        """
        Test generating non-centered patterns.

        This test checks if generating non-centered patterns results in initial generations that are not empty.
        """
        size = 10
        steps = 5
        num_samples = 1
        data = generate_data(size, steps, num_samples, GLIDER, is_centered=False)

        self.assertFalse(np.array_equal(data[0], np.zeros((size, size), dtype=int)))

    def save_plots(self, data, plot_name_prefix, output_dir):
        """
        Helper function to save generated plots.

        This function saves the generated plots to the specified output directory.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for step, grid in enumerate(data):
            plt.imshow(grid, cmap='binary')
            plt.title(f"Step {step}")
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f"{plot_name_prefix}_step_{step}.png"))
            plt.close()

if __name__ == '__main__':
    unittest.main()
