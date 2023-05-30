import numpy as np

def conjgrad(A,b,x):
    r = b - np.dot(A,x)
    p = r
    rsold = np.dot(np.transpose(r),r)
    
    for _ in range(1,len(b)+1):
        Ap = np.dot(A,p)
        alpha = rsold / np.dot(np.transpose(p),Ap)
        x = x + alpha*p
        r = r - alpha*Ap
        rsnew = np.dot(np.transpose(r),r)
        if np.sqrt(rsnew) < 1*10**(-10):
            break
        p = r + (rsnew/rsold)*p
        rsold = rsnew
    return x

def test_conjgrad():
    A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    bA = np.array([12,21,144])

    B = np.array([[2000, -577, 0], [-577, 2000, -577], [0, -577, 2000]])
    bB = np.array([144,21,12])

    x = np.array([0,0,0])
    
    print("Méthode du gradient conjugué sur A...",end="")
    xA = conjgrad(A,bA,x)
    assert(np.array_equal(np.round(np.dot(A,xA),7),bA))
    print("\tOK")

    print("Méthode du gradient conjugué sur B...",end="")
    xB = conjgrad(B,bB,x)
    assert(np.array_equal(np.round(np.dot(B,xB),7),bB))
    print("\tOK")

    return None

#test_conjgrad()

def preconditioned_conjgrad(A,b,x,M):
    r = b - np.dot(A,x)
    z = np.dot(np.linalg.inv(M),r) # z = [.,.,.]
    print(z)
    p = z
    rold = r
    for k in range(1,len(b)):
        alpha = (np.dot(np.transpose(rold),z))/(np.dot(np.dot(np.transpose(p),A),p))
        x = x + np.dot(alpha,p)
        rnew = rold - alpha*np.dot(A,p)
        print(f"renew = {rnew}")
        # Lorsque la matrice rnew est assez petite, on arrête la boucle
        if np.sqrt(np.dot(rnew,rnew)) < 1*10**(-10):
            break
        znew = np.dot(np.linalg.inv(M),rnew)
        beta = np.dot(np.transpose(rold),znew) / np.dot(np.transpose(rold),z)
        print(beta)
        p = znew + np.dot(beta,p)
        rold = rnew
        z = znew
    return x
# besoin d'un z old !!

if __name__ == "__main__":
    A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    b = np.array([12,21,144])
    x = np.array([0,0,0])
    M = np.array([[1,0,0],[0,1,0],[0,0,1]])
    x = preconditioned_conjgrad(A,b,x,M)
    print(x)
