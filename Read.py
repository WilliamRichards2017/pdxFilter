from ParseFastQ import ParseFastQ
import gzip
import time
class Read:
    def __init__(self, read):
        self.read=read
        self.base_freq = np.empty(4)
       
    def base_count(self, c):
        if c=='A':
            self.base_freq[0] +=1
        elif c == 'C':
            self.base_freq[1] +=1
        elif c == 'G':
            self.base_freq[2] +=1
        elif c=='T':
            self.base_freq[3] +=1



##file = '/uufs/chpc.utah.edu/common/home/marth-ucgdstor/projects/pdxFilter/datasets/human/blood_1.txt.gz'
file = '/uufs/chpc.utah.edu/common/home/marth-ucgdstor/projects/pdxFilter/datasets/pdx/prechemo_treated_pdx_1.txt.gz'

outfile = 'unzipped.fastq'

start_time = time.time()

inF = gzip.open(file, 'rb')
outF = open(outfile, 'wb')
outF.write( inF.read() )
inF.close()
outF.close()

print("\nTime spent unzipping: {0:.3f} min.".format((time.time() - start_time) / float(60)))

f = open('neg_sequences.txt', 'w')

from Bio import SeqIO
for record in SeqIO.parse(outfile, "fastq"):
    ##print(record.seq)
    f.write(str(record.seq)+'\n')

f.close()


print("\nTime spent filtering: {0:.3f} min.".format((time.time() - start_time) / float(60)))




'''
count = 0
reads = []
##Stores origin of DNA read sequence
read_origins = []

'''
