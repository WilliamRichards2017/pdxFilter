import csv
import numpy as np
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as pl

with open('confidence.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    right, wrong = [], []

    for row in readCSV:
        dif = float("{0:.4f}".format(abs(float(row[1][0:6])-float(row[0][0:6]))))
        if row[3] == row[4]:
            right.append(dif)
        elif row[2] == row[4]:
            wrong.append(dif)

print("right avg dif is {}".format(np.mean(right)))
print("wrong avg dif is {}\n".format(np.mean(wrong)))

rs = sorted(right)
fit = stats.norm.pdf(rs, np.mean(rs), np.std(rs))




print("50th percentile right is at {}".format(np.percentile(rs,50)))
print("60th percentile right is at {}".format(np.percentile(rs,60)))
print("70th percentile right is at {}".format(np.percentile(rs,70)))
print("80th percentile right is at {}".format(np.percentile(rs,80)))
print("90th percentile right is at {}".format(np.percentile(rs,90)))
print("95th percentile right is at {}\n".format(np.percentile(rs,95)))

pl.figure(1)
pl.subplot(221)
pl.plot(rs, fit, '-0')

pl.hist(rs, normed=True, bins=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
pl.xlim(0,15)
pl.xlabel("dif between predictions")
pl.ylabel('freq')
pl.title('correct predictions')

ws = sorted(wrong)
fit2 = stats.norm.pdf(ws, np.mean(ws), np.std(ws))

print("50th percentile wrong is at {}".format(np.percentile(ws,50)))
print("60th percentile wrong is at {}".format(np.percentile(ws,60)))
print("70th percentile wrong is at {}".format(np.percentile(ws,70)))
print("80th percentile wrong is at {}".format(np.percentile(ws,80)))
print("90th percentile wrong is at {}".format(np.percentile(ws,90)))
print("95th percentile wrong is at {}".format(np.percentile(ws,95)))



pl.subplot(222)
pl.plot(ws, fit2, '-0')
pl.hist(ws, normed=True, bins=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
pl.xlim(0,15)
pl.xlabel("dif between predictions")
pl.ylabel('freq')
pl.title('incorrect predictions')

pl.show()
