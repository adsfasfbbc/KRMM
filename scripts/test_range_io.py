import os,tempfile,unittest
from unittest.mock import patch
from range_io import contiguous_prefix,pwrite_all
class RangeIO(unittest.TestCase):
    def test_empty_file(self):
        with tempfile.TemporaryFile() as f:self.assertEqual(contiguous_prefix(f.fileno(),0),0)
    def test_sparse_tail_is_not_a_complete_prefix(self):
        with tempfile.TemporaryFile() as f:
            os.pwrite(f.fileno(),b'a'*8192,0);os.pwrite(f.fileno(),b'z'*4096,65536)
            self.assertLess(contiguous_prefix(f.fileno(),69632),69632)
            self.assertEqual(contiguous_prefix(f.fileno(),69632,True),0)
    def test_short_writes_retried(self):
        actual=os.pwrite
        with tempfile.TemporaryFile() as f,patch('range_io.os.pwrite',side_effect=lambda fd,b,o:actual(fd,b[:2],o)):
            pwrite_all(f.fileno(),b'abcdefgh',3);f.seek(3);self.assertEqual(f.read(8),b'abcdefgh')
if __name__=='__main__':unittest.main()
