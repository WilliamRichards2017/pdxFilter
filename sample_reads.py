from Bio import SeqIO
from Bio import AlignIO
from ParseFastQ import ParseFastQ
import gzip
import time
import sys
import random
files = []
for arg in sys.argv: 
    print("arg is {}".format(arg))
    files.append(arg)

start_time = time.time()

poutfile = 'good_quality_pos.fastq'
start_time = time.time()

'''inF = gzip.open(files[1], 'rb')
outF = open(poutfile, 'wb')
outF.write( inF.read() )
inF.close()
outF.close()
'''

f = open('pos.txt', 'w', buffering=100000)
i = 0

reservoir = []
t, n = 0, 4000000

for record in SeqIO.parse(poutfile, "fastq"):
    if t < n:
        reservoir.append(record.seq)
        t += 1
    else:
        m = random.randint(0,t)
        if m < n:
            reservoir[m] = record.seq

for record in reservoir:
    
    f.write(str(record)+'\n')
   


'''##Filter reads by qaulity of greater than 30
good_reads = (rec for rec in SeqIO.parse(poutfile, "fastq") 
              if min(rec.letter_annotations["phred_quality"]) >= 30)
count = SeqIO.write(good_reads, "good_quality_pos.fastq", "fastq")
print("Saved %i reads" % count)
'''
    
f.close()
print("\nTime spent sampling pos reads: {0:.3f} min.".format((time.time() - start_time) / float(60)))

start_time = time.time()


noutfile = 'good_quality_neg.fastq'
start_time = time.time()

'''inF = gzip.open(files[2], 'rb')
outF = open(noutfile, 'wb')
outF.write( inF.read() )
inF.close()
outF.close()
'''

f = open('neg.txt', 'w', buffering=100000)
i = 0

reservoir = []
t, n = 0, 4000000

for record in SeqIO.parse(noutfile, "fastq"):
    if t < n:
        reservoir.append(record.seq)
        t += 1
    else:
        m = random.randint(0,t)
        if m < n:
            reservoir[m] = record.seq

for record in reservoir:

    f.write(str(record)+'\n')


'''good_reads = (rec for rec in SeqIO.parse(noutfile, "fastq")
              if min(rec.letter_annotations["phred_quality"]) >= 20)
count = SeqIO.write(good_reads, "good_quality_neg.fastq", "fastq")
print("Saved %i reads" % count)
'''



f.close()

print("\nTime spent sampling neg: {0:.3f} min.".format((time.time() - start_time) / float(60)))
