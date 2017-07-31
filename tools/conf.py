import csv
import numpy as np
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as pl


def rel_dist(x,y):
    return abs(x-y)/(abs(x)+abs(y)/2)
    


with open('small_confidence.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    right, wrong = [], []

    for row in readCSV:
        ##dif = float("{0:.4f}".format(abs(float(row[1][0:6])-float(row[0][0:6]))))
        dif = float("{0:.4}".format(rel_dist(float(row[1][0:6]), float(row[0][0:6]))))
        if row[3] == row[4]:
            right.append(dif)
        elif row[2] == row[4]:
            wrong.append(dif)



print("right avg dif is {}".format(np.mean(right)))
print("wrong avg dif is {}\n".format(np.mean(wrong)))

print("right std dif is {}".format(np.std(right)))
print("wrong std dif is {}\n".format(np.std(wrong)))

rs = sorted(right)
fit = stats.norm.pdf(rs, np.mean(rs), np.std(rs))


print("80th percentile right is at {}".format(np.percentile(rs,80)))
print("90th percentile right is at {}".format(np.percentile(rs,90)))
print("95th percentile right is at {}".format(np.percentile(rs,95)))
print("96th percentile right is at {}".format(np.percentile(rs,96)))
print("97th percentile right is at {}".format(np.percentile(rs,97)))
print("98th percentile right is at {}".format(np.percentile(rs,98)))
print("99th percentile right is at {}\n".format(np.percentile(rs,99)))


pl.figure(1)
pl.subplot(221)
pl.plot(rs, fit, '-0')

pl.hist(rs, normed=True)
pl.xlabel("dif between predictions")
pl.ylabel('freq')
pl.title('correct predictions')

ws = sorted(wrong)
fit2 = stats.norm.pdf(ws, np.mean(ws), np.std(ws))

print("80th percentile wrong is at {}".format(np.percentile(ws,80)))
print("90th percentile wrong is at {}".format(np.percentile(ws,90)))
print("95th percentile wrong is at {}".format(np.percentile(ws,95)))
print("96th percentile wrong is at {}".format(np.percentile(ws,96)))
print("97th percentile wrong is at {}".format(np.percentile(ws,97)))
print("98th percentile wrong is at {}".format(np.percentile(ws,98)))
print("99th percentile wrong is at {}\n".format(np.percentile(ws,99)))



pl.subplot(222)
pl.plot(ws, fit2, '-0')
pl.hist(ws, normed=True)
pl.xlabel("dif between predictions")
pl.ylabel('freq')
pl.title('incorrect predictions')

pl.show()
