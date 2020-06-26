def debug(func):
    def wrapper(self, *args, **kwargs):
        print("[DEBUG]: enter {}()")
        func(self, *args)
        print(self.x)
        print("ok")
    return wrapper


class TEST(object):
    def __init__(self):
        self.x = 2

    @debug
    def say_hello(self, a, b):
        print("hello! %s" % (a+b))


test = TEST()
test.say_hello(1, 1)
