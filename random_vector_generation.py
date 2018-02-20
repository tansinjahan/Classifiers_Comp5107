import numpy as np
import matplotlib.pyplot as plt

mean_X1 = np.array([3, 1, 4])
mean2_X2 = np.array([-3,1,-4])


def generate_sigma1():
    a = 2
    b = 3
    c = 4
    alpha = 0.1
    beta = 0.2
    sigma = np.array([[a*a, beta*a*b, alpha*a*c],
              [beta*a*b, b*b, beta*b*c],
              [alpha*a*c, beta*b*c, c*c]])
    return sigma

def generate_sigma2():
    a = 2
    b = 3
    c = 4
    alpha = 0.1
    beta = 0.2

    sigma = np.array([[c * c, alpha * b * c, beta * a * c],
                       [alpha * b * c, b * b, alpha * a * b],
                       [beta * a * c, alpha * a * b, a * a]])

    return sigma


def generate_Z():
    z_vectors = np.array([])
    for i in range(0, 3):
        z = 0
        for j in range(0, 12):
            a = np.random.uniform(0, 1)
            z = z + a
        z = z - 6
        z_vectors = np.append(z_vectors,[z])

    return z_vectors

print("This is z:", generate_Z())

def generation_Of_X1():
    z_vectors = generate_Z()

    sigma1 = generate_sigma1()

    mean_X1 = np.array([3, 1, 4])

    # Eigen values and Eigen vectors of sigma1
    lambda_def1, Px1 = (np.linalg.eig(sigma1))
    lambda_def1 = np.diag(lambda_def1)
    lambda_to_the_powerHalf = np.power(lambda_def1,0.5)
    return np.dot(np.dot(Px1, lambda_to_the_powerHalf), z_vectors)+ mean_X1

def generation_Of_X2():
    z_vectors = generate_Z()

    a = 2
    b = 3
    c = 4
    alpha = 0.1
    beta = 0.2

    sigma2 = np.array([[c*c, alpha *b *c, beta *a * c],
                       [alpha*b*c, b*b, alpha* a * b],
                       [beta *a *c, alpha *a *b, a * a]])

    mean2_X2 = np.array([-3, 1, -4])

    # Eigen values and Eigen vectors of sigma1
    lambda_def2, Px2 = (np.linalg.eig(sigma2))
    lambda_def = np.diag(lambda_def2)
    lambda_to_the_powerHalf = np.power(lambda_def, 0.5)
    return np.dot(np.dot(Px2, lambda_to_the_powerHalf), z_vectors) + mean2_X2


def generate_points(M):
    X = np.array([])
    for i in range(0,5000):
        if(M == 1):
            temp = generation_Of_X1()
        elif(M == 2):
            temp = generation_Of_X2()
        X = np.append(X, temp)
    X = X.reshape(5000,3)
    print(X.shape)
    return X


print("This is 5000 points of X1:")
X1 = generate_points(1)
print (X1)
print("This is 5000 points of X2:")
X2 = generate_points(2)
print (X2)

# generation of Y1 and Y2
def generation_of_Y1():
    sigma1 = generate_sigma1()  # cov of x1
    lambda_def, Px1 = (np.linalg.eig(sigma1))  # 1st- eigval, 2nd - eigvec
    Px1_transpose = np.transpose(Px1)

    y1 = np.dot(Px1_transpose,X1.transpose())
    y1 = y1.transpose()

    # mean of Y1
    mean_of_y1 = np.dot(Px1_transpose,mean_X1)
    print("Mean of Y1:", mean_of_y1)

    # covariance of Y1
    covariance_of_Y1 = np.dot(np.dot(Px1_transpose,sigma1),Px1)
    print("Covariance of Y1:", covariance_of_Y1)

    # eigenvalues and eigenvectors of Y1

    eigenvalue_Y1, eigenvector_Y1 = np.linalg.eig(covariance_of_Y1)
    print("eigen values of  Y1:", eigenvalue_Y1)
    print("eigen vectors of Y1:", eigenvector_Y1)
    return y1


def generation_of_Y2():
    sigma1 = generate_sigma1()  # cov of x1
    lambda_def, Px1 = (np.linalg.eig(sigma1))  # 1st- eigval, 2nd - eigvec
    Px1_transpose = np.transpose(Px1)

    y2 = np.dot(Px1_transpose, X2.transpose())
    y2 = y2.transpose()

    # mean of Y2
    mean_of_y2 = np.dot(Px1_transpose,mean2_X2)
    print("Mean of Y2:", mean_of_y2)

    # covariance of Y2
    sigma2 = generate_sigma2()
    covariance_of_Y2 = np.dot(np.dot(Px1_transpose,sigma2),Px1)
    print("Covariance of Y2:", covariance_of_Y2)

    # eigenvalues and eigenvectors of Y2

    eigenvalue_Y2, eigenvector_Y2 = np.linalg.eig(covariance_of_Y2)
    print("eigen values of  Y2:", eigenvalue_Y2)
    print("eigen vectors of Y2:", eigenvector_Y2)
    return y2


print("This is 5000 points of Y1:")
Y1 = generation_of_Y1()
print (Y1)

print("This is 5000 points of Y2:")
Y2 = generation_of_Y2()
print (Y2)

# generation of Z1 and Z2
def generation_of_Z1():

    sigma1 = generate_sigma1() # cov of x1
    lambda_def, Px1 = (np.linalg.eig(sigma1)) # 1st- eigval, 2nd - eigvec
    tempZ_1 = np.power(lambda_def, -0.5)
    tempZ_1 = np.diag(tempZ_1)
    Px1_transpose = np.transpose(Px1)

    z1 = np.dot(np.dot(tempZ_1,Px1_transpose),X1.transpose())
    z1 = z1.transpose()

    # mean of Z1

    mean_of_Z1 = np.dot(np.dot(tempZ_1,Px1_transpose),mean_X1)
    print("Mean of Z1:", mean_of_Z1)

    # covariance of Z1

    #covariance_of_Z1 = np.dot(np.dot(tempZ_1,lambda_def),tempZ_1)
    covariance_of_Z1 = np.identity(3)
    print ("This is covariance of Z1:", covariance_of_Z1)

    # eigenvalues and eigenvectors of Z1

    eigenvalue_Z1, eigenvector_Z1 = np.linalg.eig(covariance_of_Z1)
    print("eigen values of  Z1:", eigenvalue_Z1)
    print("eigen vectors of Z1:", eigenvector_Z1)
    return  z1


def generation_of_Z2():

    sigma1 = generate_sigma1()  # cov of x1
    lambda_def, Px1 = (np.linalg.eig(sigma1))  # 1st- eigval, 2nd - eigvec
    tempZ_1 = np.power(lambda_def, -0.5)
    tempZ_1 = np.diag(tempZ_1)
    Px1_transpose = np.transpose(Px1)

    z2 = np.dot(np.dot(tempZ_1, Px1_transpose), X2.transpose())
    z2 = z2.transpose()

    # mean of Z2
    mean_Z2 = np.dot(np.dot(tempZ_1,Px1_transpose),mean2_X2)
    print("Mean of Z2:", mean_Z2)

    # covariance of Z2

    sigma2 = generate_sigma2()
    covariance_of_Z2 = np.dot(np.dot(np.dot(np.dot(tempZ_1,Px1_transpose),sigma2),Px1), tempZ_1)
    print("covariance of Z2:", covariance_of_Z2)

    # eigenvalues and eigenvectors of Z2
    eigenvalue_Z2, eigenvector_Z2 = np.linalg.eig(covariance_of_Z2)
    print("eigen values of Z2:", eigenvalue_Z2)
    print("eigen vectors of Z2:", eigenvector_Z2)
    return z2

print("This is 5000 points of Z1:")
Z1 = generation_of_Z1()
print (Z1)

print("This is 2000 points of Z2:")
Z2 = generation_of_Z2()
print (Z2)


# generation of V1 and V2

def generation_of_V1():

    sigma1 = generate_sigma1()  # cov of x1
    lambda_def, Px1 = (np.linalg.eig(sigma1))  # 1st- eigval, 2nd - eigvec
    tempZ_1 = np.power(lambda_def, -0.5)
    tempZ_1 = np.diag(tempZ_1)
    Px1_transpose = np.transpose(Px1)

    # covariance of Z2

    sigma2 = generate_sigma2()
    covariance_of_Z2 = np.dot(np.dot(np.dot(np.dot(tempZ_1, Px1_transpose), sigma2), Px1), tempZ_1)
    eigenvalue,Pz2 = (np.linalg.eig(covariance_of_Z2))
    Pz2_transpose = Pz2.transpose()

    v1 = np.dot(Pz2_transpose,Z1.transpose())
    v1 = v1.transpose()

    # mean and covariance of V1
    mean_of_Z1 = np.dot(np.dot(tempZ_1, Px1_transpose), mean_X1)
    mean_V1 = np.dot(Pz2_transpose,mean_of_Z1)
    print ("The mean of V1:", mean_V1)

    I = np.identity(3)

    covariance_of_V1 = np.dot(np.dot(Pz2_transpose,I),Pz2)
    print ("The covariance of V1:", covariance_of_V1)
    return v1


def generation_of_V2():

    sigma1 = generate_sigma1()  # cov of x1
    lambda_def, Px1 = (np.linalg.eig(sigma1))  # 1st- eigval, 2nd - eigvec
    tempZ_1 = np.power(lambda_def, -0.5)
    tempZ_1 = np.diag(tempZ_1)
    Px1_transpose = np.transpose(Px1)

    # covariance of Z2

    sigma2 = generate_sigma2()
    covariance_of_Z2 = np.dot(np.dot(np.dot(np.dot(tempZ_1, Px1_transpose), sigma2), Px1), tempZ_1)
    som, Pz2 = (np.linalg.eig(covariance_of_Z2))
    Pz2_transpose = Pz2.transpose()

    v2 = np.dot(Pz2_transpose, Z2.transpose())
    v2 = v2.transpose()

    # mean and covariance of V1
    mean_of_Z2 = np.dot(np.dot(tempZ_1, Px1_transpose), mean2_X2)
    mean_V2 = np.dot(Pz2_transpose, mean_of_Z2)
    print ("The Mean of V2:", mean_V2)

    covariance_of_V2 = np.dot(np.dot(Pz2_transpose, covariance_of_Z2), Pz2)
    print ("The covariance of V2:", covariance_of_V2)

    return v2


print("This is 5000 points of V1:")
V1 = generation_of_V1()
print (V1)

print("This is 5000 points of V2:")
V2 = generation_of_V2()
print (V2)


def slicing_points(X1, X2):
    X1_0_1 = X1[:,[0,1]] # X1 - X2
    X1_0_2 = X1[:,[0,2]] # X1 - X3

    X2_0_1 = X2[:,[0,1]]
    X2_0_2 = X2[:,[0,2]]

    return X1_0_1,X1_0_2,X2_0_1,X2_0_2


def generate_plot():
    X1_0_1,X1_0_2,X2_0_1,X2_0_2 = slicing_points(X1,X2)

    # To plot X1_X2 domain of X
    a = np.amax(X1)
    b = np.amin(X1)
    c = np.amax(X2)
    d = np.amin(X2)


    plt.axis([d-5,c+5,b,a])

    plt.scatter(X1_0_1[:, [0]], X1_0_1[:, [1]], c='red')
    plt.axis([d, c, b, a])
    plt.scatter(X2_0_1[:, [0]], X2_0_1[:, [1]], c='blue')
    plt.text(d+2,a-2,'red= first, blue = second')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("X1 - X2 domain")
    plt.show()

    plt.axis([d-3, c+3, b-2, a])

    plt.scatter(X1_0_2[:, 0], X1_0_2[:, 1], c='red')
    plt.scatter(X2_0_2[:, 0], X2_0_2[:, 1], c='blue')
    plt.text(d + 2, a - 2, 'red= first, blue = second')
    plt.xlabel("X1")
    plt.ylabel("X3")
    plt.title("X1 - X3 domain")
    plt.show()

    a = np.amax(Y1)
    b = np.amin(Y1)
    c = np.amax(Y2)
    d = np.amin(Y2)

    # To plot Y1_Y2 domain of Y
    Y1_0_1, Y1_0_2, Y2_0_1, Y2_0_2 = slicing_points(Y1, Y2)

    plt.axis([d - 3, c + 8, b - 6, a + 1])

    plt.scatter(Y1_0_1[:, [0]], Y1_0_1[:, [1]], c='red')
    plt.scatter(Y2_0_1[:, [0]], Y2_0_1[:, [1]], c='blue')
    plt.text(d + 2, a - 2, 'red= first, blue = second')
    plt.xlabel("Y1")
    plt.ylabel("Y2")
    plt.title("Y1 - Y2 domain")
    plt.show()

    plt.axis([d - 3, c + 9, b -2 , a])

    plt.scatter(Y1_0_2[:, 0], Y1_0_2[:, 1], c='red')
    plt.scatter(Y2_0_2[:, 0], Y2_0_2[:, 1], c='blue')
    plt.text(d + 2, a - 2, 'red= first, blue = second')
    plt.xlabel("Y1")
    plt.ylabel("Y3")
    plt.title("Y1 - Y3 domain")
    plt.show()

    a = np.amax(Z1)
    b = np.amin(Z1)
    c = np.amax(Z2)
    d = np.amin(Z2)

    Z1_0_1, Z1_0_2, Z2_0_1, Z2_0_2 = slicing_points(Z1, Z2)

    # To plot Z1_Z2 domain of Z

    plt.axis([d-3, c+3, b-5, a+2])

    plt.scatter(Z1_0_1[:, [0]], Z1_0_1[:, [1]], c='red')
    plt.scatter(Z2_0_1[:, [0]], Z2_0_1[:, [1]], c='green')
    plt.text(d + 0.1, a - 0.1, 'red= first, green = second')
    plt.xlabel("Z1")
    plt.ylabel("Z2")
    plt.title("Z1 - Z2 domain")
    plt.show()

    plt.axis([d , c , b-3, a])

    plt.scatter(Z1_0_2[:, 0], Z1_0_2[:, 1], c='red')
    plt.scatter(Z2_0_2[:, 0], Z2_0_2[:, 1], c='green')
    plt.text(d + 0.6, a - 0.6, 'red= first, green = second')
    plt.xlabel("Z1")
    plt.ylabel("Z3")
    plt.title("Z1 - Z3 domain")
    plt.show()

    a = np.amax(V1)
    b = np.amin(V1)
    c = np.amax(V2)
    d = np.amin(V2)

    V1_0_1, V1_0_2, V2_0_1, V2_0_2 = slicing_points(V1, V2)

    # To plot V1_V2 domain of V

    plt.axis([d-3 , c+0.5 , b-2, a])

    plt.scatter(V1_0_1[:, [0]], V1_0_1[:, [1]], c='red')
    plt.scatter(V2_0_1[:, [0]], V2_0_1[:, [1]], c='green')
    plt.text(d + 2, a - 0.5, 'red= first, green = second')
    plt.xlabel("V1")
    plt.ylabel("V2")
    plt.title("V1 - V2 domain")
    plt.show()

    plt.axis([d-2 , c+2 , b-3, a+3])

    plt.scatter(V1_0_2[:, 0], V1_0_2[:, 1], c='red')
    plt.scatter(V2_0_2[:, 0], V2_0_2[:, 1], c='green')
    plt.text(d + 0.2, a+2, 'red= first, green = second')
    plt.xlabel("V1")
    plt.ylabel("V3")
    plt.title("V1 - V3 domain")
    plt.show()


generate_plot()

def generate_POverall():
    sigma1 = generate_sigma1()  # cov of x1
    lambda_def, Px1 = (np.linalg.eig(sigma1))  # 1st- eigval, 2nd - eigvec
    tempZ_1 = np.power(lambda_def, -0.5)
    tempZ_1 = np.diag(tempZ_1)
    Px1_transpose = np.transpose(Px1)

    # covariance of Z2

    sigma2 = generate_sigma2()
    covariance_of_Z2 = np.dot(np.dot(np.dot(np.dot(tempZ_1, Px1_transpose), sigma2), Px1), tempZ_1)
    eigenvalue,Pz2 = (np.linalg.eig(covariance_of_Z2))
    Pz2_transpose = Pz2.transpose()

    P_overall = np.dot(np.dot(Pz2_transpose,tempZ_1),Px1_transpose)

    return P_overall

P_overall = generate_POverall()
print("This is P_overall:", P_overall)


#print("This is 2000 points of X2:", generate_points(2))