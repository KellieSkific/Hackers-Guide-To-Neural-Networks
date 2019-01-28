import numpy as np

# Base Case: Single Gate in the Circuit
# Single simple circuit with one gate
# This circuit takes two real-valued inputs and computes x * y with the * gate

x = -2
y = 3

def forwardMultiplyGate(x, y):
    return(x * y)

forwardMultiplyGate(-2, 3)

print(forwardMultiplyGate(x, y))

output = forwardMultiplyGate(x, y)

# Numerical Gradient
# Computing the derivative with respect to x

# Small tweak amount
h = 0.0001

xph = x + h
output_2 = forwardMultiplyGate(xph, y)
x_derivative = (output_2 - output) / h

# X derivative = 3.0

print "X derivative:  ", x_derivative


# Computing the derivative with respect to y
yph = y + h
output_3 = forwardMultiplyGate(x, yph)
y_derivative = (output_3 - output) / h

# Y derivative = -2.0

print "Y derivative:  ", y_derivative

# Once circuits get much more complex (e.g. entire neural networks)
# The gradient guarantees that if you have a very small step size
# You will get a higher number when you follow it's direction

step_size = 0.01
output = forwardMultiplyGate(x, y)

# Adding the step_size to our current x & y, then multiplying by the derivative of x&y
x = x + step_size * x_derivative
y = y + step_size * y_derivative
output_4 = forwardMultiplyGate(x, y)

# Output_4 = -5.8
print "Our output: ", output_4

# Analytic Gradient

x = -2
y = 3

output = forwardMultiplyGate(x, y)

# Setting x & y equal to x_gradient and y_gradient
x_gradient = y
y_gradient = x

step_size = 0.01
x += step_size * x_gradient
y += step_size * y_gradient
output_5 = forwardMultiplyGate(x, y)

# Output_5 = -5.8
print "Output once again: ", output_5


# Recursive Case: Circuits with Multiple Gates
# The expression we are computing now is f(x,y,z)=(x+y)z
# Using a & b - not to get confused with x & y

a = -2
b = 5
z = -4

# Forward Multiply Gate Function - only takes inputs a & b
def forwardMultiplyGate(a, b):
    return a * b
# Forward Add Gate Function - only takes inputs a & b
def forwardAddGate(a, b):
    return a + b
# Forward Circuit function - takes inputs a, b, and z
def forwardCircuit(a, b, z):
    q = forwardAddGate(a, b);
    f = forwardMultiplyGate(q, z)
    return f

f = forwardCircuit(a, b, z)
# f = -12
print "Forward Circuit: ", f


# BACKPROPAGATION
# Invoking the Chain Rule
# Chain Rule tells us how to combine these to get the gradient of the final output
# With respect to x & y
x = -2
y = 5
z = -4

q = forwardAddGate(a, b);
f = forwardMultiplyGate(q, z)

# Derivative of f with respect to z = q
derivative_f_wrt_z = q
# Derivative of f with respect to q = z
derivative_f_wrt_q = z

# Derivative of q with respect to both x & y is simply 1
derivative_q_wrt_x = 1.0
derivative_q_wrt_y = 1.0

# Derivative of f with respect to x = Derivative of q with respect to x * Derivative of f with respect to q
derivative_f_wrt_x = derivative_q_wrt_x * derivative_f_wrt_q

# Derivative of f with respect to y = Derivative of q with respect to y * Derivative of f with respect to q
derivative_f_wrt_y = derivative_q_wrt_y * derivative_f_wrt_q

# Gradient of f with respect to x, y, & z is equal to the Derivative of f with respect to x, y, and z
gradient_f_wrt_xyz = [derivative_f_wrt_x, derivative_f_wrt_y, derivative_f_wrt_z]

step_size = 0.01

# Add variables + step_size * derivative
x = x + step_size * derivative_f_wrt_x
y = y + step_size * derivative_f_wrt_y
z = z + step_size * derivative_f_wrt_z

q = forwardAddGate(x, y)
f = forwardMultiplyGate(q,z)

print "Q now has a higher output: ", q
print "and our circuit now gives higher output wooohooo! ", f


# Numerical Gradient Check
x = -2
y = 5
z = -4

h = 0.0001
x_derivative = (forwardCircuit(x+h,y,z) - forwardCircuit(x,y,z)) / h
y_derivative = (forwardCircuit(x,y+h,z) - forwardCircuit(x,y,z)) / h
z_derivative = (forwardCircuit(x,y,z+h) - forwardCircuit(x,y,z)) / h

print "And we should have what was computed with backprop "
print x_derivative, y_derivative, z_derivative


# Becoming a BackProp Ninja

x = a * b
dx = derivative_f_wrt_x
dq = derivative_f_wrt_q

da = b * dx
db = a * dx

x = a + b

da = 1.0 * dx
db = 1.0 * dx

q = a + b
x = q + z

dz = 1.0 * dx
dq = 1.0 * dx
da = 1.0 * dq
db = 1.0 * dq


x = a + b + z

da = 1.0 * dx
db = 1.0 * dx
dz = 1.0 * dx


x = a * b + z
da = b * dx
db = a * dx
dz = 1.0 * dx


q = a * x + b * y + z

def sigmoid(q):
    return 1.0 / (1 + np.exp(-q))
f = sigmoid(q)

df = 1.0
dq = ( f * (1 - f)) * df
da = x * dq
dx = a * dq
dy = b * dq
db = y * dq
dz = 1.0 * dq
