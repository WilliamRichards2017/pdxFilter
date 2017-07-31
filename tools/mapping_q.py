import pysam
print("plz work")
sam = pysam.AlignmengtFile("aln-pe.mapped.sorted.bam", 'rb')
i = 0 
for read in sam.fetch(until_eof=True):
    if i == 1000:
        break
    print("read quality is {}".format(pysam.mapping_quality(read)))
    i+=1


