import numpy as np

def test(fun):
    dict = {"solve": test_solve,  
            "kkt": test_kkt}
    return dict[fun.__name__](fun)

def test_solve(solve):
    H = np.array([[4, 6], [6, 9]])
    b = -np.array([14, 21])
    c = np.zeros(1)
    C = np.array([[1, -2]])
    
    x = np.array([2, 1])
    
    assert(np.allclose(solve(H, b, C, c).ravel(), x, rtol=1e-16, atol=1e-15))
    print("Test passed. Great!")
    
def test_kkt(kkt):
    H = np.array([[4, 6], [6, 9]])
    b = -np.array([14, 21])
    c = np.zeros(1)
    C = np.array([[1, -2]])
    x = np.array([2, 1])
    
    def f_fun(x):
        return x.dot(H.dot(x))/2. + x.dot(b)
    
    def c_fun(x):
        return C.dot(x) - c

    testsys = [H, b, C, c]
    systems = kkt(f_fun, c_fun, x)
    for k, outsys in enumerate(systems):
        assert(np.allclose(outsys, testsys[k], rtol=1e-5, atol=1e-5))
    print("Test passed. Great!")
    
    
