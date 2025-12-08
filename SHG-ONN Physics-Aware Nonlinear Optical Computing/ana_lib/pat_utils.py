import torch

class PATFunc:
    #f_pat = PATFunc(f_exp, mean_dt)
    def __init__(self, forward_f, backward_f):
        self.forward_f = forward_f
        self.backward_f = backward_f
        self.func = generate_func(forward_f, backward_f)
        
    def __call__(self, x):
        return self.func(x)

def generate_func(forward_f, backward_f):
    class func(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return forward_f(x)
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            torch.set_grad_enabled(True)
            y = torch.autograd.functional.vjp(backward_f, x, v=grad_output)
            torch.set_grad_enabled(False)
            return y[1]
    return func.apply