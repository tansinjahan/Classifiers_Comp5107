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
    plt.scatter(Points_X1[:], Root1[:], c='green')
    #plt.text( 2, 2, 'red= first, blue = second')
    plt.xlabel("X1")
    plt.ylabel("X3")
    plt.title("X1 - X3 domain")
    plt.show()

def a_b_c():
    inverse_sigma1 = np.linalg.inv(gn.generate_sigma1())
    inverse_sigma2 = np.linalg.inv(gn.generate_sigma2())

    a = (inverse_sigma2 - inverse_sigma1)/2

    for_b1 = np.dot(np.transpose(gn.m1),inverse_sigma1)
    for_b2 = np.dot(np.transpose(gn.m2),inverse_sigma2)

    b = for_b1 - for_b2

    for_c1 = np.log10(1)
    for_c2 = np.log10(np.linalg.det(gn.generate_sigma2())/np.linalg.det(gn.generate_sigma1()))
    #print(np.linalg.det(gn.generate_sigma2())/np.linalg.det(gn.generate_sigma1()))

    c = for_c1 +for_c2

    return a,b,c

A,B,C = a_b_c()
print("A : ", A)
print("B : ", B)
print("C : ", C)

# To print in X1-X3 domain

def discriminant_function():
    root1 = np.array([])
    root2 = np.array([])
    points_x1 = np.array([])
    for x1 in range(-15,15,1):
        p = A[2][2]
        q = ((A[0][2] * x1)+ (A[2][0] * x1) +B[2])
        r = A[0][0] * x1 *x1 + B[0] * x1 + C
        coef_array = np.array([p,q,r])
        r1, r2 = np.roots(coef_array)
        root1 = np.append(root1,r1)
        root2 = np.append(root2,r2)
        points_x1 = np.append(points_x1,[x1])

    return root1,root2,points_x1

Root1,Root2,Points_X1 = discriminant_function()
print("This is root1: " , Root1)
print("This is root2: " , Root2)
#print("This is x1:", x1)
print(Root1.shape)
print(Points_X1.shape)

generate_plot_for_X()




