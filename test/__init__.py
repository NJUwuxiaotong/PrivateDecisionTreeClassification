
def f1(x, y):
    print("%s + %s = %s" % (x, y, x+y))


def f2(func, *params):
    print("call function: %s" % func)
    func(params[0], params[1])

eval('f1')(1,2)