import csv
import numpy as np
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as pl


def rel_dist(x,y):
    return abs(x-y)/(abs(x)+abs(y)/2)
    


with open('small_conf.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    right_pos, right_neg, wrong_pos, wrong_neg = [], [], [], []

    for row in readCSV:
        ##dif = float("{0:.4f}".format(abs(float(row[1][0:6])-float(row[0][0:6]))))
        dif = float("{0:.4}".format(rel_dist(float(row[1][0:6]), float(row[0][0:6]))))
        if row[3] == row[4] and row[4] == '1.00000e+00':
            right_pos.append(dif)
        elif row[3] == row[4] and row[4] == '0.00000e+00':
            right_neg.append(dif)
        elif row[2] == row[4] and row[4] == '1.00000e+00':
            wrong_pos.append(dif)
        elif row[2] == row[4] and row[4] == '0.00000e+00':
            wrong_neg.append(dif)


'''
print("right avg dif is {}".format(np.mean(right)))
print("wrong avg dif is {}\n".format(np.mean(wrong)))

print("right std dif is {}".format(np.std(right)))
print("wrong std dif is {}\n".format(np.std(wrong)))
'''

rs_pos = sorted(right_pos)
rs_neg = sorted(right_neg)
fit1 = stats.norm.pdf(rs_pos, np.mean(rs_pos), np.std(rs_pos))
fit2 = stats.norm.pdf(rs_neg, np.mean(rs_neg), np.std(rs_neg))

'''
print("80th percentile right is at {}".format(np.percentile(rs,80)))
print("90th percentile right is at {}".format(np.percentile(rs,90)))
print("95th percentile right is at {}".format(np.percentile(rs,95)))
print("96th percentile right is at {}".format(np.percentile(rs,96)))
print("97th percentile right is at {}".format(np.percentile(rs,97)))
print("98th percentile right is at {}".format(np.percentile(rs,98)))
print("99th percentile right is at {}\n".format(np.percentile(rs,99)))
'''

pl.figure(1)

pl.subplot(221)
pl.plot(rs_pos, fit1, '-0')

pl.hist(rs_pos, normed=True)
pl.xlabel("dif between predictions")
pl.ylabel('freq')
pl.title('correct positive predictions')

pl.subplot(222)
pl.plot(rs_neg, fit2, '-0')

pl.hist(rs_neg, normed=True)
pl.xlabel("dif between predictions")
pl.ylabel('freq')
pl.title('correct negative predictions')



ws_pos = sorted(wrong_pos)
ws_neg = sorted(wrong_neg)
fit1 = stats.norm.pdf(ws_pos, np.mean(ws_pos), np.std(ws_pos))
fit2 = stats.norm.pdf(ws_neg, np.mean(ws_neg), np.std(ws_neg))

'''
print("80th percentile wrong is at {}".format(np.percentile(ws,80)))
print("90th percentile wrong is at {}".format(np.percentile(ws,90)))
print("95th percentile wrong is at {}".format(np.percentile(ws,95)))
print("96th percentile wrong is at {}".format(np.percentile(ws,96)))
print("97th percentile wrong is at {}".format(np.percentile(ws,97)))
print("98th percentile wrong is at {}".format(np.percentile(ws,98)))
print("99th percentile wrong is at {}\n".format(np.percentile(ws,99)))
'''


pl.subplot(223)
pl.plot(ws_pos, fit1, '-0')
pl.hist(ws_pos, normed=True)
pl.xlabel("dif between predictions")
pl.ylabel('freq')
pl.title('incorrect pos predictions')

pl.subplot(224)
pl.plot(ws_neg, fit2, '-0')
pl.hist(ws_neg, normed=True)
pl.xlabel("dif between predictions")
pl.ylabel('freq')
pl.title('incorrect predictions')

pl.show()
