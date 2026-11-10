# LLM Terms
### Sequence
Sequence means the full list of tokens currently in front of the model.

It grows during generation. At the start, the sequence is just the tokenized prompt. Each newly generated token is appended and becomes the next position. When the model is producing its 20th output token, the sequence is the prompt tokens plus the 19 tokens it has already generated. The KV cache is what keeps that growing sequence cheap to extend.

### Position
Position *i* is the *i*-th slot in the sequence, and each slot holds one token. In "The cat sat on the", there are 5 positions, 0 through 4, and position 1 holds "cat".

People still say "position" rather than "token" for two reasons:

- **The same token can sit at several positions.** In "the cat sat on the mat", the token "the" appears at positions 0 and 4. At the input, those two slots get identical vectors, because both are the same embedding row. After the first attention layer they've already diverged: position 4 has absorbed context from "cat sat on", and position 0 has nothing to absorb. The token is the same, but the positions carry different vectors.
- **Deep in the stack, the vector isn't really "about" its own token anymore.** As covered earlier, by the last layer the vector at position 4 is mostly a prediction of the *next* token ("mat"). Calling it "the vector at position 4" describes it more accurately than "the vector for 'the'".

A smaller point: a token isn't always a whole word. It's a subword piece from the tokenizer. Short common words like "cat" are usually one token, but a long or rare word may be split across several tokens, and therefore across several positions.

