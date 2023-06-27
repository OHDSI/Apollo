import shutil
import tempfile
import os

import unittest

import numpy as np

from simulating import cdm_data


class TestCdmData(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_folder = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_folder)

    def test_dynamic_array(self):
        x = cdm_data.DynamicArray()
        for value in range(2000):
            x.append(value)
        assert len(x) == 2000
        assert (x.collect() == np.asarray(range(2000))).all()
