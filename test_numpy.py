from numpy import linalg as LA
FILE = open("cmake-build-debug/matrix", "r")
text = FILE.read()
G = list(map(lambda x: list(map(int, x.strip().split(' '))), text.strip().split('\n')))

eigen_values = sorted(LA.eig(G)[0])
print(eigen_values)


