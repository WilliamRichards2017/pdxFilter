from Bio import SeqIO
from Bio import AlignIO
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

f = open('p.txt', 'w')
i = 0

for record in SeqIO.parse(poutfile, "fastq"):
    if i == 1000:
        break
    f.write("{},{}\n".format(record.id, record.seq))
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

f = open('n.txt', 'w')
i = 0

for record in SeqIO.parse(noutfile, "fastq"):
    if i == 1000:
        break
    f.write("{},{}\n".format(record.id, record.seq))
    i+=1


'''good_reads = (rec for rec in SeqIO.parse(noutfile, "fastq")
              if min(rec.letter_annotations["phred_quality"]) >= 20)
count = SeqIO.write(good_reads, "good_quality_neg.fastq", "fastq")
print("Saved %i reads" % count)
'''



f.close()

print("\nTime spent sampling neg: {0:.3f} min.".format((time.time() - start_time) / float(60)))
