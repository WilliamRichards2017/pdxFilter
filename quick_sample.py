from Bio import SeqIO
from Bio import AlignIO
from Bio.Align.Applications import clustalwCommandline
from ParseFastQ import ParseFastQ
import gzip
import time
import sys

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

f = open('pos_90.txt', 'w')
i = 0

for record in SeqIO.parse(poutfile, "fastq"):
    if i == 90000:
        break
    f.write(str(record.seq)+'\n')
    i+=1


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

f = open('neg_10.txt', 'w')
i = 0

for record in SeqIO.parse(noutfile, "fastq"):
    if i == 10000:
        break
    f.write(str(record.seq)+'\n')
    i+=1


'''good_reads = (rec for rec in SeqIO.parse(noutfile, "fastq")
              if min(rec.letter_annotations["phred_quality"]) >= 20)
count = SeqIO.write(good_reads, "good_quality_neg.fastq", "fastq")
print("Saved %i reads" % count)
'''



f.close()

print("\nTime spent sampling neg: {0:.3f} min.".format((time.time() - start_time) / float(60)))
