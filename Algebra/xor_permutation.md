# XOR mapping is a permutation
### XOR mapping definition
With any constant $k \in [0,\, 2^6 - 1]$, the following XOR($\oplus$) mapping is a bijection and its codomain is identical to its domain:<br/>
$f(c) = c \oplus k: c \in [0,\, 2^6 - 1] \rightarrow W$.

The codomain $W$ is determined as follows: <br/>
Since XOR never produces any carry bit and, $c$ and $k$ are represented by 6 bits, $c \oplus k$ is also represented by 6 bits. <br/>
Which means that $c \oplus k$ is also in $[0,\, 2^6 - 1]$, i.e. $W = [0,\, 2^6 - 1]$ is identical to domain.

### Proof
#### Reversible
First, we prove that $f(c)$ is reversible: $f(f(c)) = c$. <br/>
We expand $f(f(c)) = (c \oplus k) \oplus k$. Since XOR is associative, we can reassociate $(c \oplus k) \oplus k = c \oplus (k \oplus k) = c \oplus 0 = c$ <br/>
So $f(f(c)) = c$ and therefore, $f(c)$ is reversible.

#### Injective: One-to-one mapping
Next, we need to prove that $f(c)$ is **injective**, i.e. an one-to-one mapping. <br/>
Suppose we pick two any numbers $a$ and $b$ in $[0,\, 2^6 - 1]$. We need to prove that if $f(a) = f(b)$, then $a$ must be equal to $b$. <br/>
Applying $f$ to both sides, we obtain: $f(f(a)) = f(f(b))$. Since $f$ is reversible, $f(f(a)) = a$ and $f(f(b)) = b$. Therefore, $a = b$. <br/>
This means that $f(c)$ maps every $c \in [0,\, 2^6 - 1]$ to an unique output, i.e. $f(c)$ is injective.

#### Surjective: Domain-onto-codomain mapping
Since $f(c)$ is an one-to-one mapping and its domain has $2^6$ unique elements, its **range** must have $2^6$ unique elements. <br/>
In addition, its codomain $W = [0,\, 2^6 - 1]$ has exact $2^6$ unique elements. That means $f(c)$ maps its domain onto its codomain (**range** is identical to **codomain**). <br/>
Therefore, $f(c)$ is **surjective**.

Since $f(c)$ is both **injective** and **surjective**, it is a bijection. Besides, since its codomain is identical to its domain. <br/>
It is a **permutation**.
