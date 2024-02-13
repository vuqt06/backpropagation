class Value:
    """This class is used to store the value of the nod and its gradient."""
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self.backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad = 1 * out.grad
            other.grad = 1 * out.grad
        out.backward = _backward

        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out.backward = _backward

        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad = (1 - t**2) * out.grad
        out.backward = _backward

        return out