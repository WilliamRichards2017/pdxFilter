import pysam

bam = pysam.AlignmentFile('test.bam', 'rb')

for read in bam.fetch():
    print(read)

bam.close()
