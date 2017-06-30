from Bio import SeqIO
from ParseFastQ import ParseFastQ
import gzip
import time
import sys

files = []
for arg in sys.argv:
    print("arg is {}".format(arg))
    files.append(arg)

max_read = int(files[3])

outfile1 = 'file1.fastq'
outfile2 = 'file2.fastq'

# dictionary for storing the id and sequence/sequences of our reads
id_seq = {}


## we only need to unzip our fastq files once
'''
start_time = time.time()
inF1 = gzip.open(files[1], 'rb')
outF1 = open(outfile1, 'wb')
outF1.write( inF1.read() )
inF1.close()
outF1.close() 
print("\nTime spent unzipping fastq1 file: {0:.3f} min.".format((time.time() - start_time) / float(60)))
'''

## we only need to unzip our fastq fiels once
'''start_time = time.time()
inF2 = gzip.open(files[2], 'rb')
outF2 = open(outfile2, 'wb')
outF2.write( inF2.read() )
inF2.close()
outF2.close()
print("\nTime spent unzipping fastq2 file: {0:.3f} min.".format((time.time() - start_time) / float(60)))
'''


start_time = time.time()
i=0
for record in SeqIO.parse(outfile1, "fastq"):
    seq = record.seq[:max_read]
    if min(record.letter_annotations["phred_quality"]) >= 20:
        if i == 100000:
            break
        if record.id in id_seq:
            id_seq[record.id] = id_seq.get(record.id) + seq
            ##f.write(str(record.seq)+'\n')
        else:
            id_seq[record.id] = seq
        i+=1
i=0



for record in SeqIO.parse(outfile2, "fastq"):
    seq = record.seq[:max_read]
    if min(record.letter_annotations["phred_quality"]) >= 20:
        if i == 100000:
            break
        if record.id in id_seq:
            id_seq[record.id] = id_seq.get(record.id) + 'nnn' + seq
            ##print("concatinating reads to lenght: {}, count = {}".format(len(id_seq.get(record.id)),i))
            ##f.write(str(record.seq)+'\n')                                                                                                    
        else:
            id_seq[record.id] = seq
        i+=1

print("\nTime spent building dict reads: {0:.3f} min.".format((time.time() - start_time) / float(60)))


    
                    
start_time = time.time()
if files[4] == 'p':
    with open('pos.txt', 'w') as log:
        for value in id_seq.values():
            log.write('{}\n'.format(value))

elif files[4] == 'n':
    with open('neg.txt', 'w') as log:
        for value in id_seq.values():
            log.write('{}\n'.format(value))
print("\nTime writing from dict to file: {0:.3f} min.".format((time.time() - start_time) / float(60)))
