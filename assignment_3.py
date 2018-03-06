import numpy as np
import matplotlib.pyplot as plt
import random_vector_generation as gn

# For question (A)
print("This is 200 points of X1:")
X1 = gn.generate_points(1)

print("This is 200 points of X2:")
X2 = gn.generate_points(2)


mean_V1,covariance_V1,V1 = gn.generation_of_V1()
mean_V2,covariance_V2,V2 = gn.generation_of_V2()

def generate_plot_for_X(plotX1,plotX2):
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

def generate_plot_for_V(plotV1, plotV2):

    V1_0_1, V1_0_2, V2_0_1, V2_0_2 = gn.slicing_points(plotV1, plotV2)

    plt.scatter(V1_0_1[:, [0]], V1_0_1[:, [1]], c='red')
    plt.scatter(V2_0_1[:, [0]], V2_0_1[:, [1]], c='blue')
    plt.scatter(Points_V1_V2[:], Root5[:], c='green')
    plt.scatter(Points_V1_V2[:], Root6[:], c='green')
    plt.xlabel("V1")
    plt.ylabel("V2")
    plt.title("V1 - V2 domain")
    plt.show()

    plt.scatter(V1_0_2[:, 0], V1_0_2[:, 1], c='red')
    plt.scatter(V2_0_2[:, 0], V2_0_2[:, 1], c='blue')
    plt.scatter(Points_V1_V3[:], Root7[:], c='green')
    plt.scatter(Points_V1_V3[:], Root8[:], c='green')
    plt.xlabel("V1")
    plt.ylabel("V3")
    plt.title("V1 - V3 domain")
    plt.show()

def beforeDiag_a_b_c():
    inverse_sigma1 = np.linalg.inv(gn.generate_sigma1())
    inverse_sigma2 = np.linalg.inv(gn.generate_sigma2())

    a = (inverse_sigma2 - inverse_sigma1)/2

    for_b1 = np.dot(np.transpose(gn.mean_X1),inverse_sigma1)
    for_b2 = np.dot(np.transpose(gn.mean2_X2),inverse_sigma2)

    b = for_b1 - for_b2

    for_c1 = np.log(1)
    for_c2 = np.log(np.linalg.det(gn.generate_sigma2())/np.linalg.det(gn.generate_sigma1()))

    c = for_c1 +for_c2

    return a,b,c

A,B,C = beforeDiag_a_b_c()


# For question (B)
# To print in X1-X3 domain

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

Root1,Root2,Points_X1_X3 = discriminant_function_X1X3(A,B,C)


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

Root3,Root4,Points_X1_X2 = discriminant_function_X1X2(A,B,C)

generate_plot_for_X(X1,X2)
# For question (C)
true_positive_X1 = 0
true_negative_X1= 0
true_positive_V1 = 0
true_negative_V1= 0

true_positive_X2 = 0
true_negative_X2= 0
true_positive_V2 = 0
true_negative_V2= 0
def generate_classifier_x1(point,transpose_point):
    global true_positive_X1
    global true_negative_X1
    value = ((np.dot(np.dot(point, A), transpose_point)) + (np.dot(B,transpose_point)) + C)
    if value >0:
        true_positive_X1 = true_positive_X1 +1
    else:
        true_negative_X1 = true_negative_X1 + 1

    return true_positive_X1, true_negative_X1



def generate_testPoints_X1():
    test_Point_x1 = np.array([])
    for i in range(0,200):
        test_x1 = gn.generation_Of_X1()
        testx1_trans = np.transpose(test_x1)
        TP, TN = generate_classifier_x1(test_x1,testx1_trans)
        test_Point_x1 = np.append(test_Point_x1,test_x1)

    test_Point_x1 = test_Point_x1.reshape(200,3)
    v1_m,v1_c, diagonalize_x1 = gn.generation_of_V1()
    return TP,TN, test_Point_x1, v1_m,v1_c, diagonalize_x1

truePositiveX1, trueNegativeX1,testPoints_x1,mean_V1,Cov_v1, diag_V1 = generate_testPoints_X1()

def generate_classifier_x2(point,transpose_point):
    global true_positive_X2
    global true_negative_X2
    temp = np.dot(point, A)
    temp1 = np.dot(temp, transpose_point)
    temp2 = np.dot(B,transpose_point)
    value = temp1 + temp2 + C

    if value <0:
        true_positive_X2 = true_positive_X2 +1
    else:
        true_negative_X2 = true_negative_X2 + 1

    return true_positive_X2, true_negative_X2

def generate_testPoints_X2():
    test_Point_x2 = np.array([])
    for i in range(0, 200):
        test_x2 = gn.generation_Of_X2()
        testx2_trans = np.transpose(test_x2)
        TP, TN = generate_classifier_x2(test_x2,testx2_trans)
        test_Point_x2 = np.append(test_Point_x2, test_x2)
    test_Point_x2 = test_Point_x2.reshape(200, 3)
    v2_m, v2_c, diagonalize_x2 = gn.generation_of_V2()

    return TP, TN, test_Point_x2, v2_m, v2_c, diagonalize_x2


truePositiveX2, trueNegativeX2, testPoints_x2,mean_V2,Cov_V2,diag_V2 = generate_testPoints_X2()

def afterDiag_a_b_c():
    inverse_sigma1 = np.linalg.inv(Cov_v1)
    inverse_sigma2 = np.linalg.inv(Cov_V2)

    a = (inverse_sigma2 - inverse_sigma1)/2

    for_b1 = np.dot(np.transpose(mean_V1),inverse_sigma1)
    for_b2 = np.dot(np.transpose(mean_V2),inverse_sigma2)

    b = for_b1 - for_b2

    for_c1 = np.log(1)
    for_c2 = np.log(np.linalg.det(Cov_V2)/np.linalg.det(Cov_v1))

    c = for_c1 +for_c2

    return a,b,c

v_A,v_B,v_C = afterDiag_a_b_c()

def generate_classifier_v1(point):
    global true_positive_V1
    global true_negative_V1
    r,c = point.shape
    for i in range(0,r):
            p = point[i]
            p_trans =np.transpose(p)
            value = ((np.dot(np.dot(point[i], v_A), p_trans)) + (np.dot(v_B,p_trans)) + v_C)
            if value >0:
                true_positive_V1 = true_positive_V1 +1
            else:
                true_negative_V1 = true_negative_V1 + 1

    return true_positive_V1, true_negative_V1

transferDomainTP_x1, transferDomainTN_x1 = generate_classifier_v1(diag_V1)

def generate_classifier_v2(point):
    global true_positive_V2
    global true_negative_V2
    r, c = point.shape
    for i in range(0, r):
            p = point[i]
            p_trans = np.transpose(p)
            temp = np.dot(point[i], v_A)
            value = ((np.dot(temp, p_trans)) + (np.dot(v_B, p_trans)) + v_C)
            if value < 0:
                true_positive_V2 = true_positive_V2 + 1
            else:
                true_negative_V2 = true_negative_V2 + 1

    return true_positive_V2, true_negative_V2

transferDomainTP_x2,transferDomainTN_x2 = generate_classifier_v2(diag_V2)

print("truePositiveX1 -", truePositiveX1)
print("trueNegativeX1 - ",trueNegativeX1)

print("truePositiveX2 -",truePositiveX2)
print("trueNegativeX2 -",trueNegativeX2)

print("Transfer Domain truePositiveX1 -", transferDomainTP_x1)
print("Transfer Domain trueNegativeX1 - ",transferDomainTN_x1)

print("Transfer Domain truePositiveX2 -", transferDomainTP_x2)
print("Transfer Domain trueNegativeX2 - ",transferDomainTN_x2)


def accuracy(TP1,TN1,TP2,TN2):
    a = (TP1+TP2)
    b = (TP1+TN1+TP2+TN2) * 1.0
    c = float (a/b)
    c = c * 100
    return c

print("Accuracy before diagonalization of points:", accuracy(truePositiveX1,trueNegativeX1,truePositiveX2,trueNegativeX2))
print("Accuracy after diagonalization:", accuracy(transferDomainTP_x1,transferDomainTN_x1,transferDomainTP_x2,transferDomainTN_x2))


generate_plot_for_X(X1,X2)

Root5,Root6,Points_V1_V2 = discriminant_function_X1X2(v_A,v_B,v_C)
Root7,Root8,Points_V1_V3 = discriminant_function_X1X3(v_A,v_B,v_C)

generate_plot_for_V(V1, V2)
