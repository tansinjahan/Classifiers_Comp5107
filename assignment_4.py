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

# Answer to question (b)
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
    sample_points_first = className[:, 0]
    sample_points_first = np.sort(sample_points_first)
    l1 = sample_points_first.size
    sample_points_second = className[:,1]
    sample_points_second = np.sort(sample_points_second)
    l2 = sample_points_second.size
    sample_points_third = className[:, 2]
    sample_points_third = np.sort(sample_points_third)
    l3 = sample_points_third.size
    
    f_x_1 = np.array([])
    f_x_2 = np.array([])
    f_x_3 = np.array([])

    print("Please give input of which class:")
    class_type = input()
    if class_type ==1:
        print("Give input the number of points")
        points = input()
        random_X1 = gn.generate_points(1, points)
    else:
        print("Give input the number of points")
        points = input()
        random_X1 = gn.generate_points(2, points)


    random_X1_first = random_X1[:,0]
    random_X1_first = np.sort(random_X1_first)
    random_X1_second = random_X1[:,1]
    random_X1_second = np.sort(random_X1_second)
    random_X1_third = random_X1[:,2]
    random_X1_third = np.sort(random_X1_third)

    #print("random points and sample points:",random_points_first,sample_points_first)
    del_x = 0
    ex_mean = 0
    for i in random_X1_first:
        temp = 0

        for x in sample_points_first:
                temp1 = 1.0 / (np.sqrt(2 * 3.1416) * sigma)
                temp2 = np.power((i-x), 2)
                temp3 = 2 * np.power(sigma, 2)
                temp4 = np.exp(-1.0 * (temp2 / temp3))
                temp5 = temp4 * temp1
                temp = temp5 + temp
        temp = temp / l1
        del_x = temp -del_x
        ex_mean = ex_mean + del_x * temp * i
        print("This is expexted mean", ex_mean)
        f_x_1 = np.append(f_x_1, temp)

    for i in random_X1_second:
        temp = 0
        for x in sample_points_second:
            temp1 = 1.0 / (np.sqrt(2 * 3.1416) * sigma)
            temp2 = np.power((i - x), 2)
            temp3 = 2 * np.power(sigma, 2)
            temp4 = np.exp(-1.0 * (temp2 / temp3))
            temp5 = temp4 * temp1
            temp = temp5 + temp
        temp = temp / l2
        f_x_2 = np.append(f_x_2, temp)

    for i in random_X1_third:
        temp = 0
        for x in sample_points_third:
            temp1 = 1.0 / (np.sqrt(2 * 3.1416) * sigma)
            temp2 = np.power((i - x), 2)
            temp3 = 2 * np.power(sigma, 2)
            temp4 = np.exp(-1.0 * (temp2 / temp3))
            temp5 = temp4 * temp1
            temp = temp5 + temp
        temp = temp / l3
        f_x_3 = np.append(f_x_3, temp)

    return f_x_1,f_x_2,f_x_3,random_X1_first,random_X1_second,random_X1_third,random_X1

f_x_1,f_x_2,f_x_3,random_X1_first,random_X1_second,random_X1_third,random_points_X1 = parzen_window_calculate(X1,points,0.7)
f_x2_1,f_x2_2,f_x2_3,random_X2_first,random_X2_second,random_X2_third,random_points_X2 = parzen_window_calculate(X2,points,0.5)
m1_X1 = np.argmax(f_x_1)
m2_X1 = np.argmax(f_x_2)
m3_X1 = np.argmax(f_x_3)

m1_X2 = np.argmax(f_x2_1)
m2_X2 = np.argmax(f_x2_2)
m3_X2 = np.argmax(f_x2_3)

print("This is mean for first distribution of class X1",random_X1_first[m1_X1])
print("This is Mean for second distribution of class X1", random_X1_second[m2_X1])
print("This is Mean for third distribution of class X1", random_X1_third[m3_X1])

plt.scatter(random_X1_first,f_x_1)
plt.scatter(random_X1_second,f_x_2)
plt.scatter(random_X1_third,f_x_3)
plt.show()

print("This is mean for first distribution of class X2",random_X2_first[m1_X2])
print("This is Mean for second distribution of class X2", random_X2_second[m2_X2])
print("This is Mean for third distribution of class X2", random_X2_third[m3_X2])

plt.scatter(random_X2_first,f_x2_1)
plt.scatter(random_X2_second,f_x2_2)
plt.scatter(random_X2_third,f_x2_3)
plt.show()

#Answer to the question no(d)

#for ML - Optimal Bayes Discriminant
def beforeDiag_a_b_c(est_mean_X1,est_mean_X2,est_cov_X1,est_cov_X2):
    inverse_sigma1 = np.linalg.inv(est_cov_X1)
    inverse_sigma2 = np.linalg.inv(est_cov_X2)

    a = (inverse_sigma2 - inverse_sigma1)/2

    for_b1 = np.dot(np.transpose(est_mean_X1),inverse_sigma1)
    for_b2 = np.dot(np.transpose(est_mean_X2),inverse_sigma2)

    b = for_b1 - for_b2

    for_c1 = np.log(1)
    for_c2 = np.log(np.linalg.det(est_cov_X2)/np.linalg.det(est_cov_X1))

    c = for_c1 +for_c2

    return a,b,c

def discriminant_function_X1X3(A,B,C):
    root1 = np.array([])
    root2 = np.array([])
    points_x1 = np.array([])

    for x1 in np.arange(-15,10,0.1):
        #for X1 - X3 domain
        p = A[2][2]
        q = ((A[0][2] * x1)+ (A[2][0] * x1) +B[2])
        r = A[0][0] * x1 *x1 + B[0] * x1 + C

        coef_array = np.array([p,q,r])
        r1, r2 = np.roots(coef_array)

        root1 = np.append(root1,r1)
        root2 = np.append(root2,r2)
        points_x1 = np.append(points_x1,[x1])

    return root1,root2,points_x1

def discriminant_function_X1X2(A,B,C):
    root1 = np.array([])
    root2 = np.array([])
    points_x1 = np.array([])

    for x1 in np.arange(-15, 20, 0.1):
        # for X1 - X2 domain
        m = A[1][1]
        n = ((A[0][1] * x1) + (A[1][0] * x1) + B[1])
        o = A[0][0] * x1 * x1 + B[0] * x1 + C

        coef_array = np.array([m, n, o])
        r1, r2 = np.roots(coef_array)

        root1 = np.append(root1, r1)
        root2 = np.append(root2, r2)
        points_x1 = np.append(points_x1, [x1])

    return root1, root2, points_x1

ML_A,ML_B,ML_C = beforeDiag_a_b_c(est_MLmean_X1,est_MLmean_X2,est_MLcovariance_X1,est_MLcovariance_X2)
Root1_ml,Root2_ml,Points_X1_X3_ml = discriminant_function_X1X3(ML_A,ML_B,ML_C)
Root3_ml,Root4_ml,Points_X1_X2_ml = discriminant_function_X1X2(ML_A,ML_B,ML_C)

BL_A,BL_B,BL_C = beforeDiag_a_b_c(est_mean_BL_X1,est_mean_BL_X2,actual_sigma1,actual_sigma2)
Root1_Bl,Root2_Bl,Points_X1_X3_Bl = discriminant_function_X1X3(BL_A,BL_B,BL_C)
Root3_Bl,Root4_Bl,Points_X1_X2_Bl = discriminant_function_X1X2(BL_A,BL_B,BL_C)

def generate_plot_for_discriminant_func(plotX1,plotX2,Root3,Root4,Points_X1_X2,Root2,Root1,Points_X1_X3):
    X1_0_1, X1_0_2, X2_0_1, X2_0_2 = gn.slicing_points(plotX1, plotX2)

    # To plot X1_X2 domain of X

    plt.scatter(X1_0_1[:, [0]], X1_0_1[:, [1]], c='red')
    plt.scatter(X2_0_1[:, [0]], X2_0_1[:, [1]], c='blue')
    plt.scatter(Points_X1_X2[:], Root4[:], c='green')
    plt.scatter(Points_X1_X2[:], Root3[:], c='green')

    #plt.text( 2, 2, 'red= first, blue = second')
    plt.xlabel("X1")
    plt.ylabel("X2")

    plt.title("X1 - X2 domain")
    plt.show()


    plt.scatter(X1_0_2[:, 0], X1_0_2[:, 1], c='red')
    plt.scatter(X2_0_2[:, 0], X2_0_2[:, 1], c='blue')
    plt.scatter(Points_X1_X3[:], Root1[:], c='green')
    plt.scatter(Points_X1_X3[:], Root2[:], c='green')

    plt.xlabel("X1")
    plt.ylabel("X3")
    plt.title("X1 - X3 domain")
    plt.show()

generate_plot_for_discriminant_func(X1,X2,Root3_ml,Root4_ml,Points_X1_X2_ml)
generate_plot_for_discriminant_func(X1,X2,Root2_Bl,Root1_Bl,Points_X1_X3_Bl)


