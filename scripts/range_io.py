"""Conservative range-download I/O primitives; never infer completion from sparse file size."""
import os
def contiguous_prefix(fd,size,restart=False):
    if restart or size==0:return 0
    try:return min(os.lseek(fd,0,os.SEEK_HOLE),size)
    except OSError:return 0

def pwrite_all(fd,data,offset):
    consumed=0
    while consumed<len(data):
        written=os.pwrite(fd,data[consumed:],offset+consumed)
        if written<=0:raise OSError('Short pwrite')
        consumed+=written
