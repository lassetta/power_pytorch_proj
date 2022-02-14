import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
  data = np.loadtxt("results.csv", delimiter = ',')
  print(data)

  XAXIS = data[:,0]
  TOTAL = data[:,1]

  CORRECT_POSITIVE = (100. * data[:,2]) / TOTAL
  FALSE_POSITIVE = (100. * data[:,4]) / TOTAL
  FALSE_NEGATIVE = (100. * data[:,3]) / TOTAL
  CORRECT_NEGATIVE = (100. * data[:,5]) / TOTAL

  plt.figure(1)
  plt.title("Paper results")
  plt.grid()
  plt.xlabel("Approximate load level (MW)")
  plt.ylabel("Percent Totals (%)")
  plt.plot(XAXIS,CORRECT_POSITIVE, label = "correct positive")
  plt.plot(XAXIS,CORRECT_NEGATIVE, label = "correct negative")
  plt.plot(XAXIS,FALSE_POSITIVE, label = "false negative")
  plt.plot(XAXIS,FALSE_NEGATIVE, label = "false positive")
  plt.legend()
  plt.show()

