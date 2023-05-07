[link to Kaggle Notebook](https://www.kaggle.com/twoticks/scaling-forward-gradient-with-local-losses)


### Forward-mode Automatic Differentiation (AD) 
Let $f: \mathbb{R}^m \rightarrow \mathbb{R}^n$. The Jacobian of $f$, denoted by $J_f$, is a matrix of size $n \times m$. Forward-mode AD computes the matrix-vector product $J_f \boldsymbol{v}$, where $\boldsymbol{v} \in \mathbb{R}^m$. It is defined as the directional gradient along $\boldsymbol{v}$ evaluated at $x$:

$$ J_f \boldsymbol{v} \triangleq lim_{\delta \rightarrow 0} \dfrac{f(\boldsymbol{x} + \delta \boldsymbol{v}) - f(\boldsymbol{x})}{\delta} $$

In other words, forward-mode AD calculates the Jacobian-vector product of a function using a single forward pass through the computation graph, and is useful for computing gradients of high-dimensional functions.


### Weight-perturbed forward gradient:


$$g_w(w_{ij}) = (\sum_{i'j'} \nabla w_{i'j'} v_{i'j'}) v_{ij} $$

### Activity-perturbed forward gradient:


$$ g_a(w_{ij}) = x_i (\sum_{j'} \nabla z_{j'} u_{j'})u_{j} $$


### Implementation

- `jax.jvp` for computing the product of the Jacobian matrix and a vector
- `jax.grad`: used to compute the gradient of a given loss function with respect to some or all of its inputs, using automatic differentiation
- `jnp.sum`: used to compute the sum of all elements of a given tensor, which is used in various loss functions
- `jnp.mean`: used to compute the mean of all elements of a given tensor, which is used in various loss functions.
- `jnp.einsum`: used to compute the dot product of two tensors along a specific set of axes, which is used to implement the linear layers in the model
- `depthwise_conv`: used to perform a depthwise convolution of a given tensor with a set of filters, which is used to implement the convolutions in the model
- `spatial_avg_group_linear_custom_vjp`: used to perform a custom vector-Jacobian product (VJP) for the spatial average group linear layer in the model, which is necessary for computing the gradients of the loss function with respect to the weights and biases of the layer
- `spatial_avg_group_linear_cross_entropy_custom_vjp`: used to perform a custom VJP for the spatial average group linear layer when using the fused cross-entropy loss function, which is necessary for computing the gradients of the loss function with respect to the weights and biases of the layer
- `normalize`: used to normalize the activations of the model, either across channels or across spatial dimensions, which is used for activation regularization
