import numpy as np
import matplotlib.pyplot as plt
import random_vector_generation as gn

# For question (A) (generating 200 training points)
print("Please give input of how many points:")
points = input()
print("This is shape of X1:")
X1 = gn.generate_points(1,points)

print("This is shape of X2:")
X2 = gn.generate_points(2,points)

# actual/given mean and covariances for X1 and X2 training points
actual_mean1 = gn.mean_X1
print("This is mean for class1(X1):", actual_mean1)
actual_mean2 = gn.mean2_X2
print("This is mean for class2(X2):", actual_mean2)
actual_sigma1 = gn.generate_sigma1()
print("This is covariance for class1(X1):", actual_sigma1)
actual_sigma2 = gn.generate_sigma2()
print("This is covariance for class2(X2):", actual_sigma2)

#plotting 200 training points before diagonalization
def generate_plot_for_X(plotX1,plotX2):
    X1_0_1, X1_0_2, X2_0_1, X2_0_2 = gn.slicing_points(plotX1, plotX2)

    # To plot X1_X2 domain of X

    plt.scatter(X1_0_1[:, [0]], X1_0_1[:, [1]], c='red')
    plt.scatter(X2_0_1[:, [0]], X2_0_1[:, [1]], c='blue')

    plt.xlabel("X1")
    plt.ylabel("X2")

    plt.title("X1 - X2 domain")
    plt.show()
    # To plot X1_X3 domain of X
    plt.scatter(X1_0_2[:, 0], X1_0_2[:, 1], c='red')
    plt.scatter(X2_0_2[:, 0], X2_0_2[:, 1], c='blue')

    plt.xlabel("X1")
    plt.ylabel("X3")
    plt.title("X1 - X3 domain")
    plt.show()

generate_plot_for_X(X1,X2)

def estimateMean_for_ML(Name_Of_class,points):

    summation = np.array([])
    a = np.sum(Name_Of_class[:,0])
    b = np.sum(Name_Of_class[:,1])
    c = np.sum(Name_Of_class[:,2])
    summation = np.append(summation,a)
    summation = np.append(summation,b)
    summation = np.append(summation,c)
    points = float(points)
    num = np.array([points])
    mean_estimated = np.divide(summation,num)
    return mean_estimated

est_MLmean_X1 = estimateMean_for_ML(X1,points)
est_MLmean_X2 = estimateMean_for_ML(X2,points)
print("This is estimated Mean for X1 and X2:", est_MLmean_X1,est_MLmean_X2)

def estimateCovariance_forML(Name_Of_class,points,est_MLmean):
    calCov = np.array([[0.0,0.0,0.0],
                       [0.0,0.0,0.0],
                       [0.0,0.0,0.0]])
    for i in range(0,points):
        cov = Name_Of_class[i] - est_MLmean
        temp = cov
        trans_cov = temp.reshape(3,1)
        cov = cov.reshape(1,3)
        calCov = calCov + (np.dot(trans_cov,cov))
    points = float(points)
    num = np.array([points])
    cov_estimated = np.divide(calCov, num)
    return cov_estimated

est_MLcovariance_X1 = estimateCovariance_forML(X1,points,est_MLmean_X1)
est_MLcovariance_X2 = estimateCovariance_forML(X2,points,est_MLmean_X2)
print("This is estimated(ML) covariances for X1 and X2:", est_MLcovariance_X1,est_MLcovariance_X2)

def plot_convergences(array):
    plt.scatter(array[:, 0], array[:, 1], c="red")
    plt.show()

def plot_convergence_of_mean_ML(Name_Of_class,points,actual_mean):
    className = Name_Of_class
    plot = np.array([])
    for i in range(10,points+10,10):
        temp = estimateMean_for_ML(className,i)
        differ = temp - actual_mean
        norm_differ = np.linalg.norm(differ)
        array_differ = np.array([i,norm_differ])
        plot = np.append(plot,array_differ)
    plot = plot.reshape(points/10,2)
    return plot

array_Plot_ML_ConvergeMean_X1 = plot_convergence_of_mean_ML(X1,points,actual_mean1)
array_Plot_ML_ConvergeMean_X2 = plot_convergence_of_mean_ML(X2,points,actual_mean2)

plot_convergences(array_Plot_ML_ConvergeMean_X1)
plot_convergences(array_Plot_ML_ConvergeMean_X2)

def plot_convergence_of_covariance(Name_Of_class,points,actual_sigma):
    className = Name_Of_class
    plot = np.array([])
    for i in range(10, points+10, 10):
        temp = estimateMean_for_ML(className, i)
        cov = estimateCovariance_forML(className,points,temp)
        differ = cov - actual_sigma
        #print ("This is difference of covariance",differ)
        norm_differ = np.linalg.norm(differ)
        array_differ = np.array([i, norm_differ])
        plot = np.append(plot, array_differ)
    plot = plot.reshape(points / 10, 2)
    return plot

array_Plot_ML_ConvergeCovariance_X1 = plot_convergence_of_covariance(X1,points,actual_sigma1)
array_Plot_ML_ConvergeCovariance_X2 = plot_convergence_of_covariance(X2,points,actual_sigma2)

plot_convergences(array_Plot_ML_ConvergeCovariance_X1)
plot_convergences(array_Plot_ML_ConvergeCovariance_X2)

def estimateMean_for_Bayesian(Name_of_class,points):
    className = Name_of_class
    sigma_nor = np.identity(3)
    mean_nor = np.array([0.0,0.0,0.0])
    for i in range(10,points+10,10):
        var1 = (1/points) * actual_sigma1 # 1/N * sigma
        var2 = var1 + sigma_nor # 1/N * sigma + sigma_nor
        inv_var2 = np.linalg.inv(var2)
        firstPart_m_n = np.dot(np.dot(var1,inv_var2),mean_nor)
        est_Mean_ML= estimateMean_for_ML(className,points)
        secondpart_m_n = np.dot(np.dot(sigma_nor,inv_var2),est_Mean_ML)
        m_n = firstPart_m_n +secondpart_m_n

    return m_n

est_mean_BL_X1= estimateMean_for_Bayesian(X1,points)
est_mean_BL_X2= estimateMean_for_Bayesian(X2,points)
print ("this is bayesian estimated mean for X1 and X2",est_mean_BL_X1, est_mean_BL_X2)

def plot_convergence_of_mean_BL(Name_Of_class,points,actual_mean):
    className = Name_Of_class
    plot = np.array([])
    for i in range(10,points+10,10):
        temp = estimateMean_for_Bayesian(className,i)
        differ = temp - actual_mean
        norm_differ = np.linalg.norm(differ)
        array_differ = np.array([i,norm_differ])
        plot = np.append(plot,array_differ)
    plot = plot.reshape(points/10,2)
    return plot

array_Plot_BL_ConvergeMean_X1 = plot_convergence_of_mean_BL(X1,points,actual_mean1)
array_Plot_BL_ConvergeMean_X2 = plot_convergence_of_mean_BL(X2,points,actual_mean2)

plot_convergences(array_Plot_BL_ConvergeMean_X1)
plot_convergences(array_Plot_BL_ConvergeMean_X2)

#Answer of part(C)
def parzen_window_calculate(Name_of_class,points,sigma):
    className = Name_of_class;
    divisor = float(points)
    first_feature = className[:,0]
    first_feature_j = np.array([])
    second_feature = className[:,1]
    second_feature_j = np.array([])
    third_feature = className[:,2]
    third_feature_j = np.array([])
    f_x_1 = np.array([])
    f_x_2 = np.array([])
    f_x_3 = np.array([])

    for i in range(0,points,5):
        first_feature_j = np.append(first_feature_j, first_feature[i])
        second_feature_j = np.append(second_feature_j,second_feature[i])
        third_feature_j = np.append(third_feature_j,third_feature[i])

    #print("this is first feature:", first_feature_j)
    #print("this is second feature:", second_feature_j)
    #print("this is third feature:", third_feature_j)
    l1 = first_feature_j.size
    l2 = second_feature_j.size
    l3 = third_feature_j.size
    for x in first_feature_j:
        temp = 0;
        for i in range(0,points):
            if first_feature[i] != x:
                temp1 = 1.0/(np.sqrt(2*3.1416)*sigma)
                temp2 = np.power((x - first_feature[i]), 2)
                temp3 = 2*np.power(sigma,2)
                temp4 = np.exp(-1.0*(temp2/temp3))
                temp5 = temp4*temp1
                temp =temp5 +temp
            else:
                continue
        temp = temp/divisor
        f_x_1 = np.append(f_x_1, temp)

    for x in second_feature_j:
        temp = 0;
        for i in range(0, points):
            if second_feature[i] != x:
                temp1 = 1.0 / (np.sqrt(2 * 3.1416) * sigma)
                temp2 = np.power((x - second_feature[i]), 2)
                temp3 = 2 * np.power(sigma, 2)
                temp4 = np.exp(-1.0 * (temp2 / temp3))
                temp5 = temp4 * temp1
                temp = temp5 + temp
            else:
                continue
        temp = temp / divisor
        f_x_2 = np.append(f_x_2, temp)

    for x in third_feature_j:
        temp = 0;
        for i in range(0, points):
            if third_feature[i] != x:
                temp1 = 1.0 / (np.sqrt(2 * 3.1416) * sigma)
                temp2 = np.power((x - third_feature[i]), 2)
                temp3 = 2 * np.power(sigma, 2)
                temp4 = np.exp(-1.0 * (temp2 / temp3))
                temp5 = temp4 * temp1
                temp = temp5 + temp
            else:
                continue
        temp = temp / divisor
        f_x_3 = np.append(f_x_3, temp)

    return f_x_1,f_x_2,f_x_3,first_feature_j,second_feature_j,third_feature_j,dis_m1,dis_m2,dis_m3

f_x_1,f_x_2,f_x_3,first_feature_j,second_feature_j,third_feature_j,dis_m1,dis_m2,dis_m3 = parzen_window_calculate(X1,points,0.8)

print("This is Mean for first distribution", dis_m1)
print("This is Mean for second distribution", dis_m2)
print("This is Mean for third distribution", dis_m3)
plt.scatter(first_feature_j,f_x_1)
plt.show()
plt.scatter(second_feature_j,f_x_2)
plt.show()
plt.scatter(third_feature_j,f_x_3)
plt.show()





