from Bio import SeqIO
from ParseFastQ import ParseFastQ
import gzip
import time
import sys

files = []
for arg in sys.argv: 
    print("arg is {}".format(arg))
    files.append(arg)

start_time = time.time()

poutfile = 'unzipped.fastq'
start_time = time.time()
inF = gzip.open(files[1], 'rb')
outF = open(poutfile, 'wb')
outF.write( inF.read() )
inF.close()
outF.close()


f = open('pos.txt', 'w')
i = 0

for record in SeqIO.parse(poutfile, "fastq"):
    ##print(record.seq)                                                                                                                                  
    f.write(str(record.seq)+'\n')
    if i == 1000:
        break
    i+=1

f.close()
print("\nTime spent sampling pos reads: {0:.3f} min.".format((time.time() - start_time) / float(60)))

start_time = time.time()


noutfile = 'nunzipped.fastq'
start_time = time.time()
inF = gzip.open(files[2], 'rb')
outF = open(poutfile, 'wb')
outF.write( inF.read() )
inF.close()
outF.close()


f = open('neg.txt', 'w')
i = 0

for record in SeqIO.parse(noutfile, "fastq"):
    f.write(str(record.seq)+'\n')
    if i == 1000:
        break
    i+=1

f.close()

print("\nTime spent sampling neg: {0:.3f} min.".format((time.time() - start_time) / float(60)))
