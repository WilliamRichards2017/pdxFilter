'''Convert DNA sequences to One-Hot representation'''

bam = pysam.AlignmentFile('test.bam', 'rb')

iter = bam.fetch(until_eof=True)
count = 0
reads = []
##Stores origin of DNA read sequence
read_origins = []


for x in iter:
        if count < 10000:
            read = []
           ## reads.append(str(x.seq))
            for c in str(x.seq):
                if c == 'A':
                    read.append(0)
                elif c == 'C':
                    read.append(1)
                elif c == 'G':
                    read.append(2)
                else: 
                        read.append(3)
                
                ##read.append(1)
                    
            ##print(str(x.seq))
            count+=1
           ## print(read)
            reads.append(read)
        else:
            break


enc = OneHotEncoder()
##print(reads)
enc.fit(reads)

print(enc.transform(reads).toarray())

print(cosine_similarity(enc.transform(reads).toarray()[0],enc.transform(reads).toarray()[2]))
              
bam.close()
