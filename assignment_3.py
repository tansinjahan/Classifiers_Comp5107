import numpy as np
import matplotlib.pyplot as plt
import random_vector_generation as gn

print("This is 200 points of X1:")
X1 = gn.generate_points(1)
print (X1)
print("This is 200 points of X2:")
X2 = gn.generate_points(2)
print (X2)

def generate_plot_for_X():
    X1_0_1, X1_0_2, X2_0_1, X2_0_2 = gn.slicing_points(X1, X2)

    # To plot X1_X2 domain of X

    plt.scatter(X1_0_1[:, [0]], X1_0_1[:, [1]], c='red')
    plt.scatter(X2_0_1[:, [0]], X2_0_1[:, [1]], c='blue')
    #plt.text( 2, 2, 'red= first, blue = second')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("X1 - X2 domain")
    plt.show()


    plt.scatter(X1_0_2[:, 0], X1_0_2[:, 1], c='red')
    plt.scatter(X2_0_2[:, 0], X2_0_2[:, 1], c='blue')
    #plt.text( 2, 2, 'red= first, blue = second')
    plt.xlabel("X1")
    plt.ylabel("X3")
    plt.title("X1 - X3 domain")
    plt.show()

generate_plot_for_X()

def a_b_c():
    inverse_sigma1 = np.linalg.inv(gn.generate_sigma1())
    inverse_sigma2 = np.linalg.inv(gn.generate_sigma2())

    a = (inverse_sigma1 - inverse_sigma2)/2
    print("A : ", a)

    for_b1 = np.dot(np.transpose(gn.m1),inverse_sigma1)
    for_b2 = np.dot(np.transpose(gn.m2),inverse_sigma2)

    b = for_b1 - for_b2
    print("B : ", b)

    for_c1 = np.log10(1)
    for_c2 = np.log10(np.linalg.det(gn.generate_sigma2())/np.linalg.det(gn.generate_sigma1()))
    #print(np.linalg.det(gn.generate_sigma2())/np.linalg.det(gn.generate_sigma1()))

    c = for_c1 +for_c2
    print("C : ", c)

a_b_c()