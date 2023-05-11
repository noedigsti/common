# Derivative rules

### Constant rule:
If `c` is a constant, then the derivative of `c` with respect to `x` is 0: 
> $d(c) = 0$

__Example 1__: $L = 7$ then $dL = 0$


### Power rule:
For any real number `n`, the derivative of `x^n` with respect to `x` is: 
> $d(x^n) = n * x^{(n-1)}$

__Example 1__: $L = 2x^4$ then $dL = 8x^3$


### Sum/Difference rule:
The derivative of the sum or difference of two functions is the sum or difference of their derivatives: 
> $d(u + v) = du + dv$ and $d(u - v) = du - dv$

__Example 1__: $L = x^3 + 2x^2$ then $dL = 3x^2 + 4x$


### Product rule:
The derivative of the product of two functions is: 
> $d(u * v) = u * dv + v * du$

__Example 1__: $L = x^2 \cdot \sin(x)$ then $dL = 2x\sin(x) + x^2\cos(x)$
__Example 2__: $L = e^x \cdot \ln(x)$ then $dL = e^x\ln(x) + \frac{e^x}{x}$


### Quotient rule:
The derivative of the quotient of two functions is: 
> $d(u / v) = \frac{v * du - u * dv}{v^2}$
> $d(\frac{1}{v}) = \frac{-dv}{v^2}$

__Example 1__: $L = \frac{x^2}{e^x}$ then $\frac{dL}{dx} = \frac{2xe^x - x^2e^x}{e^{2x}}$
__Example 2__: $L = \frac{\ln(x)}{x^2}$ then $\frac{dL}{dx} = \frac{1 - 2\ln(x)}{x^3}$


### Chain rule:
The derivative of a composite function is the product of the derivative of the outer function and the derivative of the inner function: 
> $\frac{d(u(v))}{dx} = \frac{du}{dv} * \frac{dv}{dx}$

__Example 1__: $d(\sin(x^2)) = 2x\cos(x^2)$
__Example 2__: $d(\cos(x^3)) = -3x^2\sin(x^3)$
__Example 3__: $d(e^{\sqrt{x}}) = \frac{e^{\sqrt{x}}}{2\sqrt{x}}$


### Trigonometric derivatives: 
> $d(\sin(x)) = \cos(x)$
> $d(\cos(x)) = -\sin(x)$
> $d(\tan(x)) = 1 + \tan^2(x) = \frac{1}{\cos^2(x)}$
> $d(\cot(x)) = - (1 + \cot^2(x)) = -1\frac{1}{\sin^2(x)}$
> $\cot(x) = \frac{1}{\tan(x)}$

> $d(\sin(u)) = du.\cos(u)$
> $d(\cos(u)) = -du.\sin(u)$
> $d(\tan(u)) = du.(1 + \tan^2(u)) = \frac{du}{\cos^2(u)}$
> $d(\cot(u)) = -du.(1 + \cot^2(u)) = -\frac{du}{\sin^2(u)}$

__Example 1__: $d(\sin(2x)) = 2\cos(2x)$
__Example 2__: $d(\cos(\ln(x))) = -\frac{\sin(\ln(x))}{x}$


### Inverse trigonometric functions:
> $d(\arcsin(x)) = \frac{1}{\sqrt{1 - x^2}}$
> $d(\arccos(x)) = -\frac{1}{\sqrt{1 - x^2}}$
> $d(\arctan(x)) = \frac{1}{1 + x^2}$

__Example 1__: $\frac{d(\arcsin(\sqrt{x}))}{dx} = \frac{1}{2\sqrt{x(1-x)}}$
__Example 2__: $\frac{d(\arctan(2x))}{dx} = \frac{2}{1 + (2x)^2}$


### Exponential functions:
> $d((x^m)^n) = n * (x^m)^{(n-1)} * m * x^{(m-1)}$
> $d(e^x) = e^x$
> $d(e^{kn}) = k * e^{kn}$ (where `k` is a constant)
> $d(a^x) = a^x * \ln(a)$
> $d(a^x)^n = n * a^{(n-1)x} * a^x * \ln(a)$
> $d(u^n) = n * u^{(n-1)} * du$
> $d(\sqrt{u}) = \frac{du}{2\sqrt{u}}$

__Example 1__: $d(e^{2x}) = e^{2x}.\frac{d}{dx}[2x] = e^{2x}.2.\frac{d}{dx}[x] = 2e^{2x}.1 = 2e^{2x}$

__Example 2__: $d(3^x) = 3^x \ln(3)$


### Logarithmic functions:

> $d(\ln(x)) = \frac{1}{x}$
> $d(\ln(x)^n) = n * \ln(x)^{(n-1)} * \frac{1}{x}$ (chain rule applied)
> $d(\log_a(x)) = \frac{1}{x\ln(a)}$ (change of base formula)
> $d(\ln(u)) = \frac{du}{u}$ (chain rule applied)
> $d(\log_a(u)) = \frac{du}{u\ln(a)}$

__Example 1__: L = $\ln(x)$ then $dL = \frac{1}{x}$
__Example 2__: L = $\ln(2x)$ then $dL = \frac{1}{x}$ (chain rule applied)
__Example 3__: L = $\ln(x^2)$ then $dL = \frac{2}{x}$ (chain rule applied)