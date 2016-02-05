import arow
import unittest

class InstanceTests(unittest.TestCase):

    def testInstance1(self):
        data = "-1 1:0.1 2:0.5 9:0.1"
        feat_vec = {}
        costs = {}
        costs["neg"] = 2
        costs["pos"] = 3
        for elem in data.split()[1:]:
            fid, val = elem.split(':')
            feat_vec[fid] = float(val)
        inst = arow.Instance(feat_vec, costs)
        print inst


if __name__ == "__main__":
    unittest.main()
