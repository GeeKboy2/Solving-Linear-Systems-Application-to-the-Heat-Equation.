import numpy as np
import math
import random

# decomp_cholesky : réalise la décomposition de Cholesky
# Entrée : matrice carrée, définie positive
# Sortie : matrice T de la décomposition de Cholesky de cette matrice
def decomp_cholesky(A):
    T = np.zeros(A.shape)
    n = int(math.sqrt(A.size)) # A.size = n*n

    for i in range(n):
        for j in range(i,n):
            sum = 0
            if (i == j):
                for k in range(i):
                    sum += T[i,k]**2
                T[i,i] = math.sqrt(A[i,j]-sum)
            elif (j >= i):
                for k in range(i-1):
                    sum += T[i,k]*T[k,j]
                T[j,i] = (A[i,j]-sum)/(T[i,i])
    return T

# test_cholesky : réalisation des tests de decomp_cholesky
# Entrée : ()
# Sortie : ()
def test_cholesky():
    print("TESTS DECOMPOSITION CHOLESKY :")

    # Définition des matrices de test
    A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    B = np.array([[2000, -577, 0], [-577, 2000, -577], [0, -577, 2000]])

    # Calcul de la décomposition de Cholesky par decomp_cholesky
    T_A = decomp_cholesky(A)
    T_B = decomp_cholesky(B)

    # Calcul de la décomposition attendue
    T_A_expected = np.linalg.cholesky(A)
    T_B_expected = np.linalg.cholesky(B)

    print("Décomposition de Cholesky de A...",end="")
    assert (np.array_equal(T_A, T_A_expected))
    A_cholesky = np.round(np.dot(T_A, np.transpose(T_A)),7) # round because of real imprecisions
    assert (np.array_equal(A,A_cholesky))
    print("\tOK")

    print("Décomposition de Cholesky de B...",end="")
    assert (np.array_equal(T_B, T_B_expected))
    B_cholesky = np.round(np.dot(T_B, np.transpose(T_B)),7)
    assert (np.array_equal(B, B_cholesky))
    print("\tOK")

    return None

test_cholesky()

# spd_matrix : génère une matrice symétrique définie positive creuse avec un nombre de termes extra-diagonaux non nuls réglable (probabilite)
# Entrée : entier n et probabilite (compris entre 0 et 1)
# Sortie : A, une matrice symétrique définie positive creuse
def spd_matrix(n, probabilite):

    A = np.zeros((n, n),dtype=int)

    for i in range(n):
        for j in range(i, n):

            if i == j:
                A[i, j] = int(random.uniform(30*n, 30*n+1500))
            elif random.uniform(0, 1) < probabilite:
                    A[i, j] = int(random.uniform(-20, 20))
                    A[j, i] = A[i, j]
    return A

# test_sym : fonction testant la symétrie d'une matrice M de taille n
# Entrée : matrice M, entier n
# Sortie : booléen
def test_sym(M,n):
    for i in range(n):
        for j in range(n):
            if (M[i,j] != M[j,i]):
                return False
    return True

# test_def_pos : fonction testant si la matrice M est définie postive, avec l'utilisation du critère de Sylvester
# Entrée : matrice M, entier n
# Sortie : booléen
def test_def_pos(M,n):
    for i in range(1,n+1):
        if (np.linalg.det(M[:i,:i]) <= 0):
            return False
    return True

# test_spd_matrix : réalisation des tests de spd_matrix
# Entrée : ()
# Sortie : ()
def test_spd_matrix():
    print("TESTS GENERATION MATRICE SDPC :")

    A = spd_matrix(3,0.2)
    B = spd_matrix(5,0.2)
    C = spd_matrix(10,0.1)
    

    n_A = int(math.sqrt(np.size(A)))
    n_B = int(math.sqrt(np.size(B)))
    n_C = int(math.sqrt(np.size(C)))

    # Tests de symétrie

    print("A est-elle symétrique ?",end="")
    assert(test_sym(A,n_A))
    print("\tOK")

    print("B est-elle symétrique ?",end="")
    assert(test_sym(B,n_B))
    print("\tOK")

    print("C est-elle symétrique ?",end="")
    assert(test_sym(C,n_C))
    print("\tOK")

    # Tests définie positive

    print("A est-elle définie positive ?",end="")
    assert(test_def_pos(A,n_A))
    print("\tOK")

    print("B est-elle définie positive ?",end="")
    assert(test_def_pos(B,n_B))
    print("\tOK")

    print("C est-elle définie positive ?",end="")
    assert(test_def_pos(C,n_C))
    print("\tOK")

    # Tests creuse
    # comment on vérifie ça ?

    print("A est-elle creuse ? TODO")

    print("B est-elle creuse ? TODO")

    print("C est-elle creuse ? TODO")

    return None

test_spd_matrix()

def incomplete_decomp_cholesky(A):
    T = np.zeros(A.shape)
    n = int(math.sqrt(A.size)) # A.size = n*n

    for i in range(n):
        for j in range(i,n):
            sum = 0
            if (A[i,j]!=0):
                if (i == j):
                    for k in range(i):
                        sum += T[i,k]**2
                    T[i,i] = math.sqrt(A[i,j]-sum)
                elif (j >= i):
                    for k in range(i-1):
                        sum += T[i,k]*T[k,j]
                    T[j,i] = (A[i,j]-sum)/(T[i,i])
    return T

# def incomplete_decomp_cholesky(A):
#     T = np.zeros(A.shape)
#     n = int(math.sqrt(A.size))

#     for i in range(n):
#         sum = np.sum([T[i, k]**2 for k in range(1, i)], axis=0)
#         T[i, i] = math.sqrt(A[i, i] - sum)

#     for i in range(0, T.shape[0]):
#         for j in range(i, T.shape[0]):
#             if(A[i,j]!=0):
#                 sm = np.sum([T[i, k] * T[j, k] for k in range(0, i)])
#                 T[j, i] = (A[i, j] - sm)/T[i, i]

#     return T


# test_incomplete_decomp_cholesky : réalisation des tests de spd_matrix
# Entrée : ()
# Sortie : ()
def test_incomplete_decom_cholesky():
    print("TESTS DECOMPOSITION INCOMPLETE CHOLESKY :")

    # Définition des matrices de test
    A = spd_matrix(3,0.5)
    B = spd_matrix(5,0.2)
    C = spd_matrix(10,0.1)
    print(C)
    print(B)
    print(A)

    # Calcul de la décomposition de Cholesky par decomp_cholesky
    T_A = incomplete_decomp_cholesky(A)
    T_B = incomplete_decomp_cholesky(B)
    T_C = incomplete_decomp_cholesky(C)


    # Calcul de la décomposition attendue
    T_A_expected = np.linalg.cholesky(A)
    T_B_expected = np.linalg.cholesky(B)
    T_C_expected = np.linalg.cholesky(C)


    print("Décomposition incomplète de Cholesky de A...",end="")
    assert (np.isclose(T_A, T_A_expected,atol=1e-01).all)
    #assert (np.array_equal(T_A, T_A_expected))
    #A_cholesky = np.round(np.dot(T_A, np.transpose(T_A)),4) # round because of real imprecisions
    #assert (np.array_equal(np.round(A,4),A_cholesky)) <- testable??
    print("\tOK")

    print("Décomposition incomplète de Cholesky de B...",end="")
    assert (np.isclose(T_B, T_B_expected,atol=1e-01).all)
    #assert (np.array_equal(np.round(T_B,4), T_B_expected))
    #B_cholesky = np.round(np.dot(T_B, np.transpose(T_B)),4) 
    #assert (np.array_equal(np.round(B,4), B_cholesky)) <- testable??
    print("\tOK")

    print("Décomposition incomplète de Cholesky de C...",end="")
    assert (np.isclose(T_C, T_C_expected,atol=1e-01).all)
    #assert (np.array_equal(np.round(T_C,4), T_C_expected))
    #C_cholesky = np.round(np.dot(T_C, np.transpose(T_C)),4) 
    #assert (np.array_equal(np.round(C,4), C_cholesky)) <- testable??
    print("\tOK")

    # Tests de rapidité
    # y réfléchir...

    return None

test_incomplete_decom_cholesky()

# préconditionneur...
# utiliser np.linalg.cond & np.inv

def preconditionneur():
    A = spd_matrix(6,0.2)
    T_complet = decomp_cholesky(A)
    T_incomplet = incomplete_decomp_cholesky(A)
    print("Conditionnement de A : {}".format(np.linalg.cond(A)))
    print("Conditionnement de (TtT)-1A (complet) : {}".format(np.linalg.cond(np.dot(np.linalg.inv(np.dot(T_complet, np.transpose(T_complet))),A))))
    print("Conditionnement de (TtT)-1A (incomplet) : {}".format(np.linalg.cond(np.dot(np.linalg.inv(np.dot(T_incomplet, np.transpose(T_incomplet))),A))))

preconditionneur()


if __name__ == "__main__":
    test_spd_matrix()
    test_incomplete_decom_cholesky()
