import csv
import numpy as np
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as pl
matplotlib.use('GTK')

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
print("wrong avg dif is {}".format(np.mean(wrong)))

rs = sorted(right)
fit = stats.norm.pdf(rs, np.mean(rs), np.std(rs))

pl.figure(1)

pl.subplot(221)
pl.plot(rs, fit, '-0')
pl.hist(rs, normed=True)
pl.xlim(0,15)

ws = sorted(wrong)

fit2 = stats.norm.pdf(ws, np.mean(ws), np.std(ws))

pl.subplot(222)
pl.plot(ws, fit2, '-0')
pl.hist(ws, normed=True)
pl.xlim(0,15)

pl.show()
