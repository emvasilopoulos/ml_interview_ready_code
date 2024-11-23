import unittest

import mnn.vision.process_output.object_detection.grid as mnn_grid
import mnn.vision.image_size


class TestGrid(unittest.TestCase):

    def test_calculate_grid(self):

        image_size = mnn.vision.image_size.ImageSize(100, 100)
        grid_S = mnn.vision.image_size.ImageSize(10, 10)

        x, y = mnn_grid.calculate_grid(0.5, 0.5, image_size, grid_S)
        self.assertEqual(x, 5)
        self.assertEqual(y, 5)

        x, y = mnn_grid.calculate_grid(0.1, 0.1, image_size, grid_S)
        self.assertEqual(x, 1)
        self.assertEqual(y, 1)

        x, y = mnn_grid.calculate_grid(0.9, 0.9, image_size, grid_S)
        self.assertEqual(x, 9)
        self.assertEqual(y, 9)

    def test_calculate_grid2(self):

        image_size = mnn.vision.image_size.ImageSize(30, 30)
        grid_S = mnn.vision.image_size.ImageSize(3, 3)

        x, y = mnn_grid.calculate_grid(0.5, 0.4, image_size, grid_S)
        self.assertEqual(x, 5)
        self.assertEqual(y, 4)

        x, y = mnn_grid.calculate_grid(0.12, 0.1, image_size, grid_S)
        self.assertEqual(x, 1)
        self.assertEqual(y, 1)

        x, y = mnn_grid.calculate_grid(0.9, 0.95, image_size, grid_S)
        self.assertEqual(x, 9)
        self.assertEqual(y, 9)

    def test_twoD_grid_position_to_oneD_position(self):

        grid_S = mnn.vision.image_size.ImageSize(10, 10)

        position = mnn_grid.twoD_grid_position_to_oneD_position(5, 5, grid_S)
        self.assertEqual(position, 55)

        position = mnn_grid.twoD_grid_position_to_oneD_position(1, 1, grid_S)
        self.assertEqual(position, 11)

        position = mnn_grid.twoD_grid_position_to_oneD_position(9, 9, grid_S)
        self.assertEqual(position, 99)

    def test_oneD_position_to_twoD_grid_position(self):

        grid_S = mnn.vision.image_size.ImageSize(10, 10)

        x, y = mnn_grid.oneD_position_to_twoD_grid_position(55, grid_S)
        self.assertEqual(x, 5)
        self.assertEqual(y, 5)

        x, y = mnn_grid.oneD_position_to_twoD_grid_position(11, grid_S)
        self.assertEqual(x, 1)
        self.assertEqual(y, 1)

        x, y = mnn_grid.oneD_position_to_twoD_grid_position(99, grid_S)
        self.assertEqual(x, 9)
        self.assertEqual(y, 9)

    def test_calculate_coordinate_in_grid(self):

        image_size = mnn.vision.image_size.ImageSize(100, 100)
        grid_S = mnn.vision.image_size.ImageSize(10, 10)

        x, y = mnn_grid.calculate_coordinate_in_grid(0.5, 0.5, image_size, grid_S)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

        x, y = mnn_grid.calculate_coordinate_in_grid(0.1, 0.1, image_size, grid_S)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

        x, y = mnn_grid.calculate_coordinate_in_grid(0.9, 0.9, image_size, grid_S)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

        x, y = mnn_grid.calculate_coordinate_in_grid(0.55, 0.55, image_size, grid_S)
        self.assertAlmostEqual(x, 0.5, places=6)
        self.assertAlmostEqual(y, 0.5, places=6)

        x, y = mnn_grid.calculate_coordinate_in_grid(0.54, 0.54, image_size, grid_S)
        self.assertAlmostEqual(x, 0.4, places=6)
        self.assertAlmostEqual(y, 0.4, places=6)

    def test_calculate_real_coordinate(self):
        image_size = mnn.vision.image_size.ImageSize(100, 100)
        grid_S = mnn.vision.image_size.ImageSize(10, 10)

        x, y = mnn_grid.calculate_real_coordinate(0.5, 0.5, 5, 5, grid_S, image_size)
        self.assertEqual(x, 0.55)
        self.assertEqual(y, 0.55)


if __name__ == "__main__":
    unittest.main()
