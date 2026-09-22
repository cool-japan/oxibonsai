//! Reverse-mode gradient primitives for 1D tensors.
//!
//! Provides a lightweight [`Tensor`] type that tracks shape and optional
//! gradients, together with a set of element-wise and reduction operations.
//! Manual, hand-derived reverse-mode backward formulas live in the
//! [`backward`] sub-module so that a training loop can orchestrate its own
//! backward pass without depending on an external autodiff framework.
//!
//! **This is not forward-mode automatic differentiation.** There are no
//! dual numbers and no tangent propagation through the forward pass here —
//! `backward::*` are closed-form gradient formulas written by hand for each
//! operation (e.g. `sigmoid_backward`, `linear_backward`), which a caller
//! composes manually. That is reverse-mode-*style* manual backpropagation,
//! not forward-mode AD; an earlier version of this doc comment described it
//! as the latter, which was incorrect.
//!
//! # Design
//!
//! - Tensors store data as flat `Vec<f32>` in row-major order.
//! - Operations always produce *new* tensors; no aliasing occurs.
//! - Gradients are accumulated with [`Tensor::accumulate_grad`] and cleared
//!   with [`Tensor::zero_grad`].
//! - Shape/length mismatches on caller-supplied tensors are reported via
//!   [`GradientError`], never via `panic!`/`assert!` — see
//!   [`Tensor::new`], [`Tensor::add`], [`Tensor::mul`], [`Tensor::matmul`],
//!   and [`Tensor::accumulate_grad`].
//!
//! ## Reachability
//!
//! As of this writing, nothing else in the crate constructs a [`Tensor`]
//! from this module or calls into [`backward`] — there is no training loop
//! wired up to consume these primitives yet. This module is a tested,
//! documented primitive for that future work, not a wired feature.

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors from [`Tensor`]'s checked element-wise and matrix operations.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GradientError {
    /// [`Tensor::add`] or [`Tensor::mul`] was called on tensors with
    /// different shapes.
    #[error("{op}: shape mismatch ({a:?} vs {b:?})")]
    ShapeMismatch {
        /// Name of the operation that failed (e.g. `"add"`, `"mul"`).
        op: &'static str,
        /// Shape of the left-hand tensor.
        a: Vec<usize>,
        /// Shape of the right-hand tensor.
        b: Vec<usize>,
    },
    /// [`Tensor::accumulate_grad`] was called with a gradient slice whose
    /// length does not match the tensor's own data length.
    #[error("accumulate_grad: gradient length {got} does not match tensor length {expected}")]
    GradLengthMismatch {
        /// The tensor's own element count.
        expected: usize,
        /// The length of the gradient slice that was passed in.
        got: usize,
    },
    /// [`Tensor::matmul`]'s left-hand operand did not have `m * k` elements.
    #[error("matmul: left operand has {got} elements, expected m*k = {expected}")]
    LeftOperandSize {
        /// `m * k`, the element count the left operand was expected to have.
        expected: usize,
        /// The left operand's actual element count.
        got: usize,
    },
    /// [`Tensor::matmul`]'s right-hand operand did not have `k * n` elements.
    #[error("matmul: right operand has {got} elements, expected k*n = {expected}")]
    RightOperandSize {
        /// `k * n`, the element count the right operand was expected to have.
        expected: usize,
        /// The right operand's actual element count.
        got: usize,
    },
    /// [`Tensor::new`] was called with a `data` vector whose length does not
    /// match the product of the dimensions in `shape`.
    #[error(
        "Tensor::new: data has {data_len} elements, but shape {shape:?} needs {shape_product}"
    )]
    NewLengthMismatch {
        /// `data.len()` as supplied by the caller.
        data_len: usize,
        /// The `shape` as supplied by the caller.
        shape: Vec<usize>,
        /// `shape.iter().product::<usize>()`.
        shape_product: usize,
    },
}

// ─── Tensor ──────────────────────────────────────────────────────────────────

/// A value that tracks its gradient during the backward pass.
///
/// Data is stored flat (row-major). The `shape` field records the logical
/// dimensions; `data.len()` must equal `shape.iter().product()`.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Flat, row-major storage.
    pub data: Vec<f32>,
    /// Accumulated gradient (same shape as `data`). `None` until the first
    /// call to [`Tensor::accumulate_grad`] or [`Tensor::zero_grad`].
    pub grad: Option<Vec<f32>>,
    /// Whether this tensor participates in gradient computation.
    pub requires_grad: bool,
    shape: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor from flat data and a shape.
    ///
    /// # Errors
    ///
    /// Returns [`GradientError::NewLengthMismatch`] if `data.len() !=
    /// shape.iter().product::<usize>()` — never panics.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, GradientError> {
        let shape_product = shape.iter().product::<usize>();
        if data.len() != shape_product {
            return Err(GradientError::NewLengthMismatch {
                data_len: data.len(),
                shape,
                shape_product,
            });
        }
        Ok(Self {
            data,
            grad: None,
            requires_grad: false,
            shape,
        })
    }

    /// Enable gradient tracking for this tensor (builder pattern).
    pub fn requires_grad(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    /// Create a zero-filled tensor with the given shape.
    pub fn zeros(shape: &[usize]) -> Self {
        let n = shape.iter().product();
        Self::new(vec![0.0f32; n], shape.to_vec())
            .expect("freshly allocated data length matches shape product by construction")
    }

    /// Create a one-filled tensor with the given shape.
    pub fn ones(shape: &[usize]) -> Self {
        let n = shape.iter().product();
        Self::new(vec![1.0f32; n], shape.to_vec())
            .expect("freshly allocated data length matches shape product by construction")
    }

    /// Create a scalar tensor (shape `[1]`) from a single value.
    pub fn from_scalar(v: f32) -> Self {
        Self::new(vec![v], vec![1]).expect("single-element data matches shape [1] by construction")
    }

    /// Logical shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Zero the accumulated gradient in-place.
    ///
    /// If no gradient buffer exists yet, one is allocated and filled with
    /// zeros.
    pub fn zero_grad(&mut self) {
        match self.grad.as_mut() {
            Some(g) => g.iter_mut().for_each(|x| *x = 0.0),
            None => self.grad = Some(vec![0.0f32; self.data.len()]),
        }
    }

    /// Accumulate `grad` into `self.grad` element-wise (`self.grad += grad`).
    ///
    /// Allocates the gradient buffer if it does not exist yet.
    ///
    /// # Errors
    ///
    /// Returns [`GradientError::GradLengthMismatch`] if
    /// `grad.len() != self.data.len()` — never panics.
    pub fn accumulate_grad(&mut self, grad: &[f32]) -> Result<(), GradientError> {
        if grad.len() != self.data.len() {
            return Err(GradientError::GradLengthMismatch {
                expected: self.data.len(),
                got: grad.len(),
            });
        }
        match self.grad.as_mut() {
            Some(g) => {
                for (dst, src) in g.iter_mut().zip(grad.iter()) {
                    *dst += src;
                }
            }
            None => {
                self.grad = Some(grad.to_vec());
            }
        }
        Ok(())
    }

    /// Return a copy of this tensor with `requires_grad = false` and no
    /// gradient buffer — suitable for feeding into loss calculations that
    /// should not propagate further.
    pub fn detach(&self) -> Self {
        Self {
            data: self.data.clone(),
            grad: None,
            requires_grad: false,
            shape: self.shape.clone(),
        }
    }

    // ── Basic operations ─────────────────────────────────────────────────────

    /// Element-wise addition.  Shapes must match.
    ///
    /// # Errors
    ///
    /// Returns [`GradientError::ShapeMismatch`] if `self.shape() !=
    /// other.shape()` — never panics.
    pub fn add(&self, other: &Tensor) -> Result<Tensor, GradientError> {
        if self.shape != other.shape {
            return Err(GradientError::ShapeMismatch {
                op: "add",
                a: self.shape.clone(),
                b: other.shape.clone(),
            });
        }
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Element-wise multiplication.  Shapes must match.
    ///
    /// # Errors
    ///
    /// Returns [`GradientError::ShapeMismatch`] if `self.shape() !=
    /// other.shape()` — never panics.
    pub fn mul(&self, other: &Tensor) -> Result<Tensor, GradientError> {
        if self.shape != other.shape {
            return Err(GradientError::ShapeMismatch {
                op: "mul",
                a: self.shape.clone(),
                b: other.shape.clone(),
            });
        }
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Matrix multiplication: `self` (m×k) @ `other` (k×n) → (m×n).
    ///
    /// Both tensors are treated as flat row-major matrices.
    ///
    /// # Errors
    ///
    /// Returns [`GradientError::LeftOperandSize`] /
    /// [`GradientError::RightOperandSize`] if `self`/`other` do not have
    /// `m*k`/`k*n` elements respectively — never panics.
    pub fn matmul(
        &self,
        other: &Tensor,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Tensor, GradientError> {
        if self.data.len() != m * k {
            return Err(GradientError::LeftOperandSize {
                expected: m * k,
                got: self.data.len(),
            });
        }
        if other.data.len() != k * n {
            return Err(GradientError::RightOperandSize {
                expected: k * n,
                got: other.data.len(),
            });
        }
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += self.data[i * k + l] * other.data[l * n + j];
                }
                out[i * n + j] = sum;
            }
        }
        Tensor::new(out, vec![m, n])
    }

    /// Element-wise ReLU: `max(0, x)`.
    pub fn relu(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x.max(0.0)).collect();
        Tensor::new(data, self.shape.clone())
            .expect("element-wise map preserves length; shape is unchanged from a valid tensor")
    }

    /// Element-wise sigmoid: `1 / (1 + exp(-x))`.
    pub fn sigmoid(&self) -> Tensor {
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        Tensor::new(data, self.shape.clone())
            .expect("element-wise map preserves length; shape is unchanged from a valid tensor")
    }

    /// Softmax along the last dimension.
    ///
    /// For a 1-D or flat tensor the entire tensor is treated as one
    /// probability distribution.
    pub fn softmax(&self) -> Tensor {
        // Determine the stride of the last axis.
        let last_dim = self.shape.last().copied().unwrap_or(self.data.len());
        let batch = self.data.len() / last_dim.max(1);
        let mut data = self.data.clone();
        for b in 0..batch {
            let start = b * last_dim;
            let slice = &mut data[start..start + last_dim];
            // Numerically stable: subtract max before exp.
            let max_val = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in slice.iter_mut() {
                *v = (*v - max_val).exp();
                sum += *v;
            }
            if sum > 0.0 {
                for v in slice.iter_mut() {
                    *v /= sum;
                }
            }
        }
        Tensor::new(data, self.shape.clone())
            .expect("softmax preserves length; shape is unchanged from a valid tensor")
    }

    /// Reduce to scalar mean.
    pub fn mean(&self) -> Tensor {
        let n = self.data.len() as f32;
        let m = self.data.iter().sum::<f32>() / n.max(1.0);
        Tensor::from_scalar(m)
    }

    /// Reduce to scalar sum.
    pub fn sum(&self) -> Tensor {
        let s = self.data.iter().sum::<f32>();
        Tensor::from_scalar(s)
    }

    /// Element-wise negation.
    pub fn neg(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| -x).collect();
        Tensor::new(data, self.shape.clone())
            .expect("element-wise map preserves length; shape is unchanged from a valid tensor")
    }

    /// Element-wise natural logarithm.  Values ≤ 0 produce `-inf` / `NaN`
    /// as per IEEE 754; callers are responsible for ensuring positivity.
    pub fn log(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x.ln()).collect();
        Tensor::new(data, self.shape.clone())
            .expect("element-wise map preserves length; shape is unchanged from a valid tensor")
    }
}

// ─── Backward helpers ────────────────────────────────────────────────────────

/// Manual backward-pass gradient formulas for common operations.
///
/// These functions compute *input* gradients given the upstream gradient
/// (`grad_output`) and the saved forward-pass values. They do **not** update
/// any tensor in-place; callers should pass the result to
/// [`Tensor::accumulate_grad`].
pub mod backward {
    use super::Tensor;

    /// Gradient of `sum(x)` w.r.t. `x`: a constant `1` broadcast to every
    /// element.
    pub fn sum_backward(grad_output: f32, input: &Tensor) -> Vec<f32> {
        vec![grad_output; input.data.len()]
    }

    /// Gradient of `mean(x)` w.r.t. `x`: `1/n` broadcast to every element.
    pub fn mean_backward(grad_output: f32, input: &Tensor) -> Vec<f32> {
        let n = input.data.len() as f32;
        let scale = grad_output / n.max(1.0);
        vec![scale; input.data.len()]
    }

    /// Gradient of `ReLU(x)` w.r.t. `x`:
    /// `grad_output[i]` if `input[i] > 0`, otherwise `0`.
    pub fn relu_backward(grad_output: &[f32], input: &Tensor) -> Vec<f32> {
        input
            .data
            .iter()
            .zip(grad_output.iter())
            .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
            .collect()
    }

    /// Gradient of `sigmoid(x)` w.r.t. `x`:
    /// `grad_output[i] * σ(x)[i] * (1 - σ(x)[i])`.
    ///
    /// `output` should be the *result* of the forward sigmoid call (already
    /// in (0, 1)), which avoids recomputing the expensive exp.
    pub fn sigmoid_backward(grad_output: &[f32], output: &Tensor) -> Vec<f32> {
        output
            .data
            .iter()
            .zip(grad_output.iter())
            .map(|(&s, &g)| g * s * (1.0 - s))
            .collect()
    }

    /// Gradients of a linear layer `y = x W + b`.
    ///
    /// - `grad_output`: upstream gradient `dL/dy`, shape `[m, n]`
    /// - `input`: activations `x`, shape `[m, k]`
    /// - `weights`: weight matrix `W`, shape `[k, n]` (row-major)
    ///
    /// Returns `(dL/dx, dL/dW, dL/db)` with shapes `[m,k]`, `[k,n]`, `[n]`.
    pub fn linear_backward(
        grad_output: &[f32],
        input: &[f32],
        weights: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        // dL/dx = grad_output @ W^T   →  shape [m, k]
        let mut dx = vec![0.0f32; m * k];
        for i in 0..m {
            for j in 0..k {
                let mut s = 0.0f32;
                for l in 0..n {
                    s += grad_output[i * n + l] * weights[j * n + l];
                }
                dx[i * k + j] = s;
            }
        }

        // dL/dW = x^T @ grad_output  →  shape [k, n]
        let mut dw = vec![0.0f32; k * n];
        for i in 0..k {
            for j in 0..n {
                let mut s = 0.0f32;
                for l in 0..m {
                    s += input[l * k + i] * grad_output[l * n + j];
                }
                dw[i * n + j] = s;
            }
        }

        // dL/db = sum over batch dim   →  shape [n]
        let mut db = vec![0.0f32; n];
        for i in 0..m {
            for j in 0..n {
                db[j] += grad_output[i * n + j];
            }
        }

        (dx, dw, db)
    }

    /// Combined softmax + NLL cross-entropy backward.
    ///
    /// For a single token with logits of shape `[vocab_size]` and a one-hot
    /// target `target_id`, the gradient of the cross-entropy loss w.r.t. the
    /// logits is simply `softmax(logits) - one_hot(target_id)`.
    pub fn cross_entropy_backward(logits: &[f32], target_id: u32) -> Vec<f32> {
        // Compute softmax.
        let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = probs.iter().sum();
        let inv_sum = if sum > 0.0 { 1.0 / sum } else { 1.0 };
        for p in probs.iter_mut() {
            *p *= inv_sum;
        }
        // Subtract one-hot.
        let idx = target_id as usize;
        if idx < probs.len() {
            probs[idx] -= 1.0;
        }
        probs
    }
}

// ─── Loss functions ───────────────────────────────────────────────────────────

/// Cross-entropy loss for a single token prediction.
///
/// `loss = -log(softmax(logits)[target_id])`
pub fn cross_entropy_loss(logits: &[f32], target_id: u32) -> f32 {
    // Numerically stable softmax log-probability.
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let idx = target_id as usize;
    if idx >= logits.len() || sum == 0.0 {
        return f32::INFINITY;
    }
    let log_prob = exps[idx].ln() - sum.ln();
    -log_prob
}

/// Perplexity over a sequence of `(logits, target_id)` pairs.
///
/// `perplexity = exp(mean(-log p(target_i)))`
pub fn sequence_perplexity(pairs: &[(Vec<f32>, u32)]) -> f32 {
    if pairs.is_empty() {
        return f32::INFINITY;
    }
    let avg_nll: f32 = pairs
        .iter()
        .map(|(logits, target)| cross_entropy_loss(logits, *target))
        .sum::<f32>()
        / pairs.len() as f32;
    avg_nll.exp()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_tensor_add() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).expect("valid tensor");
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]).expect("valid tensor");
        let c = a.add(&b).expect("matching shapes must succeed");
        assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_tensor_add_shape_mismatch_returns_err() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).expect("valid tensor");
        let b = Tensor::new(vec![1.0, 2.0], vec![2]).expect("valid tensor");
        let err = a
            .add(&b)
            .expect_err("mismatched shapes must be rejected, not panic");
        assert!(matches!(
            err,
            GradientError::ShapeMismatch { op: "add", .. }
        ));
    }

    #[test]
    fn test_tensor_mul() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).expect("valid tensor");
        let b = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]).expect("valid tensor");
        let c = a.mul(&b).expect("matching shapes must succeed");
        assert_eq!(c.data, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_tensor_mul_shape_mismatch_returns_err() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).expect("valid tensor");
        let b = Tensor::new(vec![1.0, 2.0], vec![2]).expect("valid tensor");
        let err = a
            .mul(&b)
            .expect_err("mismatched shapes must be rejected, not panic");
        assert!(matches!(
            err,
            GradientError::ShapeMismatch { op: "mul", .. }
        ));
    }

    #[test]
    fn test_tensor_matmul() {
        // 2×3 @ 3×2 = 2×2
        // [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).expect("valid tensor");
        let b =
            Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).expect("valid tensor");
        let c = a
            .matmul(&b, 2, 3, 2)
            .expect("shapes are consistent with m,k,n");
        // Row 0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // Row 1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert!(approx_eq(c.data[0], 58.0));
        assert!(approx_eq(c.data[1], 64.0));
        assert!(approx_eq(c.data[2], 139.0));
        assert!(approx_eq(c.data[3], 154.0));
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn test_tensor_matmul_left_operand_size_mismatch_returns_err() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).expect("valid tensor"); // 3 elements, but m*k=2*3=6
        let b = Tensor::new(vec![0.0; 6], vec![3, 2]).expect("valid tensor");
        let err = a
            .matmul(&b, 2, 3, 2)
            .expect_err("left operand size mismatch must be rejected, not panic");
        assert!(matches!(
            err,
            GradientError::LeftOperandSize {
                expected: 6,
                got: 3
            }
        ));
    }

    #[test]
    fn test_tensor_matmul_right_operand_size_mismatch_returns_err() {
        let a = Tensor::new(vec![0.0; 6], vec![2, 3]).expect("valid tensor");
        let b = Tensor::new(vec![1.0, 2.0], vec![2]).expect("valid tensor"); // 2 elements, but k*n=3*2=6
        let err = a
            .matmul(&b, 2, 3, 2)
            .expect_err("right operand size mismatch must be rejected, not panic");
        assert!(matches!(
            err,
            GradientError::RightOperandSize {
                expected: 6,
                got: 2
            }
        ));
    }

    #[test]
    fn test_tensor_relu_forward() {
        let t = Tensor::new(vec![-2.0, -0.5, 0.0, 0.5, 2.0], vec![5]).expect("valid tensor");
        let r = t.relu();
        assert_eq!(r.data, vec![0.0, 0.0, 0.0, 0.5, 2.0]);
    }

    #[test]
    fn test_tensor_sigmoid_forward() {
        let t = Tensor::from_scalar(0.0);
        let s = t.sigmoid();
        assert!(approx_eq(s.data[0], 0.5), "sigmoid(0) must be 0.5");

        let large = Tensor::from_scalar(100.0);
        let sl = large.sigmoid();
        assert!(sl.data[0] > 0.99, "sigmoid(100) must be close to 1");
    }

    #[test]
    fn test_tensor_softmax_sums_to_one() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).expect("valid tensor");
        let s = t.softmax();
        let total: f32 = s.data.iter().sum();
        assert!(approx_eq(total, 1.0), "softmax must sum to 1, got {total}");
        for &p in &s.data {
            assert!(
                (0.0..=1.0).contains(&p),
                "each probability must be in [0,1]"
            );
        }
    }

    #[test]
    fn test_tensor_mean_scalar() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).expect("valid tensor");
        let m = t.mean();
        assert_eq!(m.shape(), &[1]);
        assert!(approx_eq(m.data[0], 2.5));
    }

    #[test]
    fn test_relu_backward_zeros_negatives() {
        let input = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![4]).expect("valid tensor");
        let grad_out = vec![1.0f32; 4];
        let grad_in = backward::relu_backward(&grad_out, &input);
        assert_eq!(grad_in, vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sigmoid_backward_shape() {
        let output = Tensor::new(vec![0.5f32; 4], vec![4]).expect("valid tensor");
        let grad_out = vec![1.0f32; 4];
        let grad_in = backward::sigmoid_backward(&grad_out, &output);
        assert_eq!(grad_in.len(), 4);
        // σ(x)(1-σ(x)) at σ=0.5 → 0.25
        for &g in &grad_in {
            assert!(approx_eq(g, 0.25), "expected 0.25, got {g}");
        }
    }

    #[test]
    fn test_linear_backward_shapes() {
        // m=2, k=3, n=4
        let grad_output = vec![1.0f32; 2 * 4];
        let input = vec![0.5f32; 2 * 3];
        let weights = vec![0.1f32; 3 * 4];
        let (dx, dw, db) = backward::linear_backward(&grad_output, &input, &weights, 2, 3, 4);
        assert_eq!(dx.len(), 2 * 3, "dL/dx shape mismatch");
        assert_eq!(dw.len(), 3 * 4, "dL/dW shape mismatch");
        assert_eq!(db.len(), 4, "dL/db shape mismatch");
    }

    #[test]
    fn test_cross_entropy_loss_basic() {
        // Logits heavily favour class 0.
        let logits = vec![10.0f32, 0.0, 0.0];
        let loss_correct = cross_entropy_loss(&logits, 0);
        let loss_wrong = cross_entropy_loss(&logits, 1);
        assert!(loss_correct < 0.01, "loss for correct class must be near 0");
        assert!(loss_wrong > 5.0, "loss for wrong class must be high");
    }

    #[test]
    fn test_cross_entropy_backward_sums_to_zero() {
        // The gradient softmax(logits) - one_hot(target) sums to 0.
        let logits = vec![1.0f32, 2.0, 3.0];
        let grad = backward::cross_entropy_backward(&logits, 1);
        let total: f32 = grad.iter().sum();
        assert!(
            total.abs() < EPS,
            "cross-entropy gradient must sum to ~0, got {total}"
        );
    }

    #[test]
    fn test_sequence_perplexity() {
        // Perfect predictor: logits heavily favour target class each time.
        let pairs: Vec<(Vec<f32>, u32)> = (0..3)
            .map(|i| {
                let mut l = vec![0.0f32; 4];
                l[i] = 20.0;
                (l, i as u32)
            })
            .collect();
        let ppl = sequence_perplexity(&pairs);
        assert!(
            ppl < 1.01,
            "near-perfect predictor must have perplexity ~1, got {ppl}"
        );

        // Empty should return infinity.
        assert!(sequence_perplexity(&[]).is_infinite());
    }

    #[test]
    fn test_tensor_accumulate_grad() {
        let mut t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])
            .expect("valid tensor")
            .requires_grad();
        t.accumulate_grad(&[0.1, 0.2, 0.3])
            .expect("matching length must succeed");
        t.accumulate_grad(&[0.1, 0.2, 0.3])
            .expect("matching length must succeed");
        let grad = t.grad.as_ref().expect("grad must be Some");
        assert!(approx_eq(grad[0], 0.2));
        assert!(approx_eq(grad[1], 0.4));
        assert!(approx_eq(grad[2], 0.6));
    }

    #[test]
    fn test_tensor_accumulate_grad_length_mismatch_returns_err() {
        let mut t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3])
            .expect("valid tensor")
            .requires_grad();
        let err = t
            .accumulate_grad(&[0.1, 0.2])
            .expect_err("length mismatch must be rejected, not panic");
        assert!(matches!(
            err,
            GradientError::GradLengthMismatch {
                expected: 3,
                got: 2
            }
        ));
    }

    #[test]
    fn test_tensor_new_length_mismatch_returns_err_not_panic() {
        // 4 elements of data, but shape [3] only accounts for 3 — this used
        // to be a debug-only `debug_assert_eq!` panic on caller input; it
        // must now be a checked `Err`, in both debug and release builds.
        let err = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![3])
            .expect_err("length/shape mismatch must be rejected, not panic");
        assert!(matches!(
            err,
            GradientError::NewLengthMismatch {
                data_len: 4,
                shape_product: 3,
                ..
            }
        ));
    }
}
