import arow
import unittest

class InstanceTests(unittest.TestCase):

    def testInstance1(self):
        data = "-1 1:0.1 2:0.5 9:0.1"
        inst = arow.instance_from_svm_input(data)
        print inst


class AROWTests(unittest.TestCase):

    def testAROW1(self):
        pass
        

if __name__ == "__main__":
    unittest.main()
