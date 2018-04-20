

class Test:
    def __init__(self, input_):
        self.input = input_
        self.input_2 = None

    @classmethod
    def do_things(cls, input_):
        cls(input_)

    def do_things_2(self, input_):
        self.input_2 = input_


Test.do_things(input_=3)

test = Test(input_=0)

test.input      # None
test.input_2    # None
