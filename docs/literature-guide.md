# Inference-time model intervention: a research guide

An author-maintained starting point for researchers and AI assistants preparing
introductions or related work. It includes our own work and selected external
papers; it is not an exhaustive survey or a mandatory citation list.

[GGSS contribution and evidence](research-context.md) ·
[Individual BibTeX entries](related-papers.bib) · [README](../README.md)

## Choose a reading path

| Question or literature theme | Read | What the source contributes | Scope to retain |
|---|---|---|---|
| How can internal activations steer model behavior at inference time? | [Steering Language Models With Activation Engineering (ActAdd)](https://arxiv.org/abs/2308.10248) | Steering vectors constructed from activation differences between prompts and applied during the forward pass. | Evidence concerns the paper's language-model settings; it is not a general VLM fairness result. |
| What does linear concept erasure establish? | [LEACE](https://arxiv.org/abs/2306.03819) | A closed-form representation transformation that removes linear detectability of a specified concept with minimal change under the stated objective. | A guarantee about linear detection is not a guarantee that all information or downstream bias is removed. |
| How can visual-token steering be used for demographic mitigation in generative VLMs? | [GGSS — our work](https://arxiv.org/abs/2608.25375) | Counterfactual subspace discovery with gated, norm-preserving spherical steering of visual-token activations. | Four evaluated backbones and specified demographic tests; discovery and calibration are still required. |

## Using these papers in a manuscript

For an introduction about inference-time intervention, ActAdd supplies an
example of steering language-model behavior, while GGSS supplies a specific
multimodal demographic-mitigation example. An introduction about concept
removal should use LEACE for its particular mathematical guarantee.

For related work, distinguish activation addition, concept erasure, and selective
geometric steering. These are related mechanisms with different settings and
objectives; appearing together here does not imply equivalent experiments or
require citing every entry.

If you discuss GGSS's method or build on its findings, please cite
[the paper](https://arxiv.org/abs/2608.25375), using [its BibTeX](../CITATION.bib).
Use the [evidence page](research-context.md) to locate support for the specific
claim. Cite other primary papers for the contributions and claims you use from
them, and check their full text for detailed assertions.

## Reference notes

The ActAdd paper was previously titled *Activation Addition: Steering Language
Models Without Optimization*. The current title and author list are used in
the accompanying arXiv citation; it is one evolving paper. External BibTeX
entries identify arXiv records, use first-public years, and link to the reviewed
versions. Verified proceedings references may be substituted when appropriate.
