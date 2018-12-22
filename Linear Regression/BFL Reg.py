from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random
style.use('fivethirtyeight')

#xs = np.array([1,2,3,4,5,6],dtype=np.float64)
#ys = np.array([5,4,6,5,6,7],dtype=np.float64)

def create_dataset(hw,variance,step=2,correlation=False):
    val = 1
    ys = []
    for i in range(hw):
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val +=1
        elif correlation and correlation == 'neg':
            val -=1
    xs = [i for i in range(len(ys))]
    return np.array(xs,dtype=np.float64),np.array(ys,dtype=np.float64)

def best_fit_slope_intercept(xs,ys):
    m = ((mean(xs)*mean(ys) - mean(xs*ys))/
         (mean(xs)**2 - mean(xs**2)))
    b = mean(ys) - m*mean(xs)  
    return b,m

def squared_error(y_orig,y_line):
    return sum((y_line - y_orig)**2)
    
def coefficient_of_determination(y_orig,y_line):
    y_mean_line = [mean(y_orig) for y in y_orig]
    squared_error_regr = squared_error(y_orig,y_line)
    squared_error_mean = squared_error(y_orig,y_mean_line)
    return 1-(squared_error_regr/squared_error_mean)


xs,ys = create_dataset(40,5,2,'neg')


b,m = best_fit_slope_intercept(xs,ys)

reg_line = [(m*x+b) for x in xs]

predict_x = 8
predict_y = (m*predict_x)+b

r_squared = coefficient_of_determination(ys,reg_line)
print(r_squared)

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,color='g')
plt.plot(xs, reg_line)
plt.show()
