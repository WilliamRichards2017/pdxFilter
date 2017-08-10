import csv
import argparse
import time
from Bio import SeqIO
from Bio import AlignIO
from ParseFastQ import ParseFastQ
import numpy as np

parser = argparse.ArgumentParser(description='filter reads classified as mouse that are above a certain bayesian derived confidence level')
parser.add_argument('-conf',  help="confidence threshold, defined as the bayesian probability that a given prediction is correct given the prediction output values", type=float)
args = parser.parse_args()


confidence_threshold = .9

if args.conf:
    confidence_threshold = float(args.conf)

confidence = 'confidence2.csv'
raw_fastq = 'fastqs/good_quality_neg.fastq'

def rel_dist(x,y):
    return abs(x-y)/(abs(x)+abs(y)/2)

def prob_right_given_dif(right_hist, total_hist, dif):
    bin_number = int(dif//0.05)
    if total_hist[bin_number] == 0:
        return 1
    return right_hist[bin_number]//total_hist[bin_number]

start_time = time.time()
with open(confidence) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')

    right_pos, right_neg, wrong_pos, wrong_neg = [], [], [], []

    for row in readCSV:
        dif = float("{0:.4}".format(rel_dist(float(row[2][0:6]), float(row[3][0:6]))))
        
        if row[3] == row[4] and row[4] == '1.00000e+00':
            right_pos.append(dif)
        elif row[3] == row[4] and row[4] == '0.00000e+00':
            right_neg.append(dif)
        elif row[2] == row[4] and row[4] == '1.00000e+00':
            wrong_pos.append(dif)
        elif row[2] == row[4] and row[4] == '0.00000e+00':
            wrong_neg.append(dif)
        
rp_hist, rp_edges = np.histogram(
    right_pos,
    bins=40,
    range=(0, 2),
    density=False)

wp_hist, wp_edges = np.histogram(
    wrong_pos,
    bins=40,
    range=(0,2),
    density=False)

rn_hist, rn_edges = np.histogram(
    right_neg,
    bins=40,
    range=(0,2),
    density=False)

wn_hist, wn_edges = np.histogram(
    wrong_neg,
    bins=40,
    range=(0,2),
    density=False)

right_hist = rp_hist + rp_hist
wrong_hist = wp_hist + wn_hist
total_hist = right_hist+wrong_hist

with open('/uufs/chpc.utah.edu/common/home/u0401321/classifier/runs/1501259215/checkpoints/predictions.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    reads_to_filter = []

    for row in readCSV:
        dif = float("{0:.4}".format(rel_dist(float(row[2][0:6]), float(row[3][0:6]))))
        if prob_right_given_dif(right_hist,total_hist,dif) >= confidence_threshold:
            reads_to_filter.append(row[0])
            
    

outfile = 'filtered_reads.fastq'

for record in SeqIO.parse(raw_fastq, "fastq"):
    if record.id not in reads_to_filter:
        SeqIO.write(record, outfile, "fastq")
    else:
        print("filtered out read {}".format(record.id))
        
print("\nTime spent filtering reads: {0:.3f} min.".format((time.time() - start_time) / float(60)))
