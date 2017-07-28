import csv
import time
from Bio import SeqIO
from Bio import AlignIO
from ParseFastQ import ParseFastQ


confidence_threshold = 0.3
raw_fastq = 'good_quality_neg.fastq'

def rel_dist(x,y):
    return abs(x-y)/(abs(x)+abs(y)/2)


start_time = time.time()
with open('/uufs/chpc.utah.edu/common/home/u0401321/classifier/runs/1501259215/checkpoints/predictions.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')

    reads_to_filter = []

    for row in readCSV:
        dif = float("{0:.4}".format(rel_dist(float(row[2][0:6]), float(row[3][0:6]))))
	## if read is mouse and we are confident about it, append id
        if row[1] == '0.0' and dif > confidence_threshold:
            reads_to_filter.append(row[0])

outfile = 'filtered_reads.fastq'

for record in SeqIO.parse(raw_fastq, "fastq"):
    if record.id not in reads_to_filter:
        SeqIO.write(record, outfile, "fastq")
    else:
        print("filtered out read {}".format(record.id))
        
print("\nTime spent filtering reads: {0:.3f} min.".format((time.time() - start_time) / float(60)))
