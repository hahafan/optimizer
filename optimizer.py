# sgd:


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.key():
            params[key] = -self.lr * grads[key]


# Momentum:
#add momentum on sgd
class Momentum:
    def __init__(self, lr=0.01, momentum=0.95):
        self.lr = lr
        self.momentum = momentum
        self.previous = {}
        for key in params.key():
            self.previous[key] = np.zeros_like(params[key])

    def update(self, params, grads):
        for key in params.key():
            self.previous[key] = self.momentum * self.previous[
                key] - self.lr * grads[key]
            params[key] += self.previous[key]


#Nestrov
#update momentum and the gradient of momentum
class Nestrov:
    def __init__(self, lr=0.01, momentum):
        self.lr = lr
        self.momentum = momentum
        self.previous = {}
        for key in params.key():
            self.previous[key] = np.zeros_like(params[key])

    def update(self, params, grads):
        for key in params.key():
            self.previous[key] = self.momentum * self.previous[
                key] - self.lr * grads[key]
            params[key] += self.momentum * self.previous[
                key] - self.lr * grads[key]


#AdaGrad
#adaptable, update learning according history gradient
#firstly big update, finally slow update
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.his = {}
        for key in params.key():
            self.his[key] = np.zeros_like(params[key])

    def update(self, params, grads):
        for key in params.key():
            self.his[key] += np.square(grads[key])
            params[key] -= self.lr * grads[key] / (np.sqrt(self.his[key]) +
                                                   1e-8)


#the learning rate of adagrad will decrease continuely,RMSprop is used to improve this drawback
class RMSprop:
    def __init__(self, lr=0.01, decay_rate=0.95):
        self.lr = lr
        self.decay_rate = decay_rate
        self.his = {}
        for key in params.key():
            self.his[key] = np.zeros_like(params[key])

    def update(self, params, grads):
        for key in params.key():
            self.his[key] *= self.decay_rate
            self.his[key] += (1 - self.decay_rate) * np.squre(grads[key])
            params[key] -= self.lr * grads[key] / (np.sqrt(self.his[key]) +
                                                   1e-8)


#adam
#combine momentum and adaptive


class adam:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.step = 0
        self.m = {}
        self.v = {}
        for key in params.key():
            self.m[key] = np.zeros_like(params[key])
            self.v[key] = np.zeros_like(params[key])

    def update(self, params, grads):
        self.step += 1
        lr_n = self.lr * np.sqrt(1.0 - self.beta2**self.step) / (
            1.0 - self.beta1**self.step)
        for key in params.key():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            params[key] -= lr_n * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
