import numpy as np


def pageMat(mat):
    return mat.T / np.sum(mat, axis=1)


def pageRank(mat):
    e_vals, e_vecs = np.linalg.eig(pageMat(mat))
    myVec = np.abs(e_vecs[:, 0].real)
    sumVec = np.sum(myVec)
    pageVec = myVec / sumVec
    indices = np.argsort(pageVec)[::-1]
    print("Web Id:", indices + 1)
    print("PageRank:", np.round(pageVec[indices], 3))
    return indices, pageVec


def calcHITS(mat):
    A = np.dot(mat.T, mat)
    H = np.dot(mat, mat.T)
    eA_vals, eA_vecs = np.linalg.eig(A)
    eH_vals, eH_vecs = np.linalg.eig(H)
    myVecA = np.abs(eA_vecs[:, 0].real)
    myVecH = np.abs(eH_vecs[:, 0].real)
    indicesA = np.argsort(myVecA)[::-1]
    print("Authority Id:", indicesA + 1)
    print("Authority:", np.round(myVecA[indicesA], 3))
    indicesH = np.argsort(myVecH)[::-1]
    print("Hub Id:", indicesH + 1)
    print("Hub:", np.round(myVecH[indicesH], 3))
    return ((indicesA, myVecA), (indicesH, myVecH))


if __name__ == "__main__":
    np.set_printoptions(precision=3, floatmode="fixed")
    A = np.array(
        [
            [0, 1, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 1, 0],
            [1, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [0, 1, 0, 1, 0, 1, 0],
        ]
    )
    print(A)
    print(pageMat(A))
    pageRank(A)
    calcHITS(A)
