from Bio import SeqIO
from Bio import AlignIO
from ParseFastQ import ParseFastQ
import gzip
import time
import sys
import random
import argparse 

## Define optional command line arguments
parser = argparse.ArgumentParser(description='randomly sample N reads from fastq file(s), optional quality filter')
parser.add_argument('-N', nargs=1, help="number of reads to sample from file(s)", type=int)
parser.add_argument('-s', nargs=1, help="sample reads from a single fastq file writes out to unknown.txt")
parser.add_argument('-d', nargs=2, help="sample reads from a positive and negative fastqx files, writes out to pos.txt and neg.txt respectively")
parser.add_argument('-q', nargs=1, help="minimum_phread_quality for all bases in a read for quality filter")
args = parser.parse_args()


## Read number of reads to sample, if provided
## Default number of reads to sample: 40000000
if args.N:
    n = args.N
else:
    n = 40000000

## Handles quality filtering if two known files are provided
if args.q and args.d:
    poutfile, noutfile = d[0], d[1]
    q =args.q

    ## Qualiity filtering for postive data file
    good_reads = (rec for rec in SeqIO.parse(poutfile, "fastq")
                  if min(rec.letter_annotations["phred_quality"]) >= q)
    count = SeqIO.write(good_reads, "good_quality_pos.fastq", "fastq")  

    ##  Quality filtering for negative data file 
    good_reads = (rec for rec in SeqIO.parse(noutfile, "fastq")
                  if min(rec.letter_annotations["phred_quality"]) >= q)
    count = SeqIO.write(good_reads, "good_quality_neg.fastq", "fastq")                                                                                                          

## Handles quality filtering for unknown file
if args.q and args.s:
    uoutfile = args.s
    q = args.q
    
    good_reads = (rec for rec in SeqIO.parse(uoutfile, "fastq")
                  if min(rec.letter_annotations["phred_quality"]) >= q)
    count = SeqIO.write(good_reads, "good_quality_unknown.fastq", "fastq")

## Handles sampling of multiple known samples
if args.d:
    poutfile, noutfile = d[0], d[1]
    ## Use the quality filtered fastqs if provided
    if args.q:
        poutfile, noutfile = 'good_quality_pos.txt', 'good_quality_neg.txt'

    #Begin resevoir sampling of positive reads
    f = open('pos.txt', 'w', buffering=100000)
    i, t = 0, 0
    reservoir = []
    
    for record in SeqIO.parse(poutfile, "fastq"):
        if t < n:
            reservoir.append("{},{}".format(record.id, record.seq))
            t += 1
        else:
            m = random.randint(0,t)
            if m < n:
                reservoir[m] = "{},{}".format(record.id, record.seq)
    for record in reservoir:
        f.write(str(record)+'\n')

    f.close()
    print("\nTime spent sampling pos reads: {0:.3f} min.".format((time.time() - start_time) / float(60)))

    #begin resevoir sampling of negative reads
    f = open('neg.txt', 'w', buffering=100000)
    i, t = 0, 0
    
    reservoir = []
    
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

## Handles sampling for a single unknown data file
if args.s:
    uoutfile = args.s
    if args.q:
       uoutfile  = 'good_quality_unknown.txt'

    #Begin resevoir sampling of unknown reads                                                                                                                                   
    f = open('unknown.txt', 'w', buffering=100000)
    i, t = 0, 0
    reservoir = []

    for record in SeqIO.parse(uoutfile, "fastq"):
        if t < n:
            reservoir.append("{},{}".format(record.id, record.seq))
            t += 1
        else:
            m = random.randint(0,t)
            if m < n:
                reservoir[m] = "{},{}".format(record.id, record.seq)
    for record in reservoir:
        f.write(str(record)+'\n')

    f.close()
    print("\nTime spent sampling unknown reads: {0:.3f} min.".format((time.time() - start_time) / float(60)))


    


