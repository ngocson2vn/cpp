# Direct Sums
In formal linear algebra, there are actually **two** distinct but related definitions of a direct sum: the **Internal Direct Sum** and the **External Direct Sum**.

Most introductory linear algebra textbooks teach the *internal* direct sum first, which involves adding vectors together.

Here is the exact difference between the two and how they connect.

### 1. The Internal Direct Sum (What you likely saw)

The internal direct sum applies when you already have a large parent vector space $V$, and you are looking at two **subspaces** inside it, $U$ and $W$.

Because $U$ and $W$ live inside the same parent space $V$, their vectors *do* have the same dimension (the dimension of $V$), and you *can* use standard vector addition on them.

The sum $V = U \oplus W$ is called a direct sum if every vector $v \in V$ can be written **uniquely** as:


$v = u + w$


where $u \in U$ and $w \in W$. In this definition, the $\oplus$ symbol represents literal vector addition of elements from two subspaces that share only the zero vector ($U \cap W = \{0\}$).

### 2. The External Direct Sum

The external direct sum is used when you have two completely independent, separate vector spaces, $X$ and $Y$, which might have completely different dimensions. They do not share a parent space.

To combine them, you define a brand new vector space $Z = X \oplus Y$ (sometimes written as the Cartesian product $X \times Y$).
The elements of this new space are ordered pairs:


$(x, y)$


where $x \in X$ and $y \in Y$. Here, you are not doing vector addition between $x$ and $y$. You are binding them together into a new mathematical object.

### The Bridge: Why they use the same symbol ($\oplus$)

In abstract algebra, the internal and external direct sums are **isomorphic**—they are mathematically identical in structure.

If we take our external direct sum of independent spaces $X$ and $Y$ (like $V_{\text{msg}}$ and $V_{\text{block}}$), the elements are pairs like $(\vec{m}, \vec{b})$.

We can split this pair apart using vector addition within the new, larger space:


$(\vec{m}, \vec{b}) = (\vec{m}, \vec{0}) + (\vec{0}, \vec{b})$

Now look at those two pieces:

1. The set of all vectors looking like $(\vec{m}, \vec{0})$ forms a subspace inside the new space. Let's call it $\tilde{X}$.
2. The set of all vectors looking like $(\vec{0}, \vec{b})$ forms another subspace. Let's call it $\tilde{Y}$.

Now, because both $\tilde{X}$ and $\tilde{Y}$ live inside the same parent space, we can add their vectors together. And because they only overlap at $(\vec{0}, \vec{0})$, we have arrived exactly at the definition of the **Internal Direct Sum**:


$(\vec{m}, \vec{b}) = \tilde{x} + \tilde{y}$

