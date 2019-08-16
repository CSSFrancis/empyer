import unittest
from CusterOperations.seqtoHDF5 import loadHeader, loadMRCfile,saveFile


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = '/media/hdd/home/FEM/DenseSampling/'
        self.filename = '16-24-03.211.seq'
        self.darkrefFile = '/media/hdd/home/FEM/DenseSampling/16-24-03.211.seq.dark.mrc'
        self.gainrefFile ='/media/hdd/home/FEM/DenseSampling/16-24-03.211.seq.gain.mrc'

    def test_read_header(self):
        filename ='/media/hdd/home/FEM/DenseSampling/16-47-57.638.seq'
        darkrefFile = '/media/hdd/home/FEM/DenseSampling/16-47-57.638.seq.dark.mrc'
        darkref = loadMRCfile(darkrefFile)
        nframes, head = loadHeader(filename, darkref)
        print(nframes, head)

    def test_save(self):
        dark = loadMRCfile(self.darkrefFile)
        gain = loadMRCfile(self.gainrefFile)
        numframes, true_imagesize = loadHeader(self.directory+self.filename, dark)
        saveFile(directory='/media/hdd/home/FEM/DenseSampling/',
                 filename=self.filename,
                 darkref=dark,
                 gainref=gain,
                 numframes=numframes,
                 true_imagesize=true_imagesize,
                 savename='test')

if __name__ == '__main__':
    unittest.main()
