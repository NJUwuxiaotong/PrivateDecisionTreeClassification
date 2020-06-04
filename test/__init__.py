def f1(x, y, *params):
    f2(2,2, params)

def f2(x, y, *params):
    f3(x, y, params[0])

def f3(x, y, *params):
    print(params[0])

f2(1,1,1)