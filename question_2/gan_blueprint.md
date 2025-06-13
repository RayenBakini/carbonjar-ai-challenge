# GAN Design – Generate Synthetic Emissions Data

## Why GAN?

We want to generate fake but realistic emissions data to help with missing data or simulations.
Since this is **tabular data** (not images), we need a simple GAN adapted to numbers.

---

## Generator (G)

**Goal:** Take random noise and turn it into something that looks like a real emissions record.

**Architecture:**

- Input: random noise vector (e.g. size 32)
- Dense layer (64) → ReLU
- Dense layer (128) → ReLU
- Output layer: same number of features as the real dataset → `sigmoid` or `tanh` depending on scaling


---

## Discriminator (D)

**Goal:** Say if a data row is real or fake.

**Architecture:**

- Input: one data row (same number of features)
- Dense layer (128) → LeakyReLU
- Dense layer (64) → LeakyReLU
- Output layer: 1 neuron → `sigmoid` (probability real/fake)

*LeakyReLU is used here to keep learning active even when values are negative. `sigmoid` gives a score between 0 and 1.*

---

## Training

- I would train both networks in turns (D, then G)
- Use **Binary Cross-Entropy** as the loss
- Normalize the data before feeding it into the networks (e.g. with MinMaxScaler)

---

## Why it fits our data

- Emissions data is not an image → no convolution
- Dense layers are better for numerical columns
- The structure is simple and works well for tabular data


