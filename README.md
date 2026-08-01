# VoicePrint

**A forensic framework for speaker disentanglement in AI-generated voice blends — without the bullshit.**

---

## TL;DR

VoicePrint is a research toolbox that answers one question: **"Who is in there?"**

You feed it an audio file — maybe a voice clone, maybe a synthetic blend — and a set of reference voice samples. VoicePrint tells you **which voices were used to create it** , and **how strongly each one contributed**.

This is not a speaker identification system. This is **source attribution for synthetic speech**.

---

## The Problem — In Plain German

You have a voice. It sounds like a person. But it's not — it's a clone, stitched together from five different real voices by some black-box AI. The question: **Which ones?**

Voice cloning is trivial now. You upload five MP3s to MiniMax, ElevenLabs, or any other service, and you get a new, unique voice that sounds like none of them — but is built from all of them.

**The catch:** There is no way to tell which original voices were used. The AI doesn't tell you. The vendor doesn't tell you. The law doesn't know how to handle it.

This is a problem for:

- **Voice actors** who want to know if their voice was used without consent.
- **Lawyers** who need technical evidence for copyright or personality rights cases.
- **Researchers** who want to understand what these models actually learn.

VoicePrint is my answer to that.

---

## What VoicePrint Does (and Doesn't)

**What it does:**

- Takes a **target audio file** (the voice clone / blend)
- Takes a **set of reference MP3s** (the suspected source voices)
- Extracts **speaker embeddings** from both
- Computes **cosine similarity** between the target and each reference
- Outputs a **ranking** — who is in there, and how much

**What it doesn't do:**

- Claim 100% accuracy — because that would be a lie.
- Work on every clone model equally well — because some models distort more than others.
- Replace a forensic expert — it's a tool, not a witness.

**What it is:**

- A **reproducible, open-source method** for a new problem.
- A **baseline** for further research.
- A **transparent** alternative to commercial black-box solutions.

---

## How It Works (The Short Version)

1.  **Extraction**: VoicePrint uses a pre-trained speaker embedding model (ECAPA-TDNN, WavLM-TDNN, or Resemblyzer) to turn every audio file into a **vector** — a mathematical fingerprint of the voice.

2.  **Comparison**: It calculates the **cosine similarity** between the target vector and each reference vector. The result is a number between 0 and 1 — higher means more similar.

3.  **Interpretation**: VoicePrint doesn't just give you numbers. It gives you a **ranking**, a **confidence estimate**, and (in future versions) a **visualization** of how the blend was composed.

That's it. No black magic. Just linear algebra and a bit of signal processing.

---

## The Reality Check (Read This)

**This will never be 100% accurate.** Not because my code is bad, but because the problem is fundamentally underdetermined.

Voice cloning models **do not** preserve the original embeddings. They **transform, interpolate, and warp** them into a new space. The resulting voice is a **new point in that space** — not a linear combination of the inputs.

**What this means for you:**

- A similarity of 0.85 does not mean "85% of this voice is Speaker X."
- It means: "In the embedding space, this clone is closer to Speaker X than to Speaker Y."
- The ranking (X > Y > Z) is more reliable than the absolute numbers.

**What this means for the project:**

- We need **systematic experiments** to understand how different models affect the embeddings.
- We need **multiple comparison methods** — not just cosine similarity.
- We need to be **honest about the limitations**.

This is a research project, not a product.

---

## Planned Features

| Feature | Status | Notes |
|---|---|---|
| **Speaker Embedding Extraction** | Planned | ECAPA-TDNN, WavLM-TDNN, Resemblyzer |
| **Cosine Similarity Comparison** | Planned | Baseline method |
| **Multi-Model Voting** | Planned | Combine results from multiple embedding models |
| **Target Speaker Extraction** | Planned | Isolate the contribution of a single source voice from the blend |
| **Batch Experiment Mode** | Planned | Run systematic experiments with controlled parameters |
| **GUI (Gradio/Streamlit)** | Planned | Drag-and-drop, visual results |
| **Exportable Reports** | Planned | PDF or HTML output for forensic use |
| **Explainable Results (Vo-Ve)** | Research | Attribute-based similarity explanation |

---

## The Ethics of This

VoicePrint is a **dual-use technology**. It can be used to:

- **Protect** voice actors from unauthorized use.
- **Expose** synthetic media and deepfakes.

It can also be misused to:

- **Identify** speakers in anonymized data.
- **Reverse-engineer** voice cloning systems.

**I am aware of this.** That's why I'm releasing the code — not the models, not the datasets. The code is the method. The method is transparent. The method can be audited.

I believe that **transparency is the best defense** against misuse. If we don't understand how these systems work, we can't regulate them.

---

## The Open Source Approach

Unlike MouthMind, VoicePrint **is open source**.

Why? Because:

1.  **Transparency matters**. If this is going to be used in legal contexts, the method must be auditable.
2.  **The field needs baselines**. Right now, there are no open-source tools for this specific problem.
3.  **I want to be proven wrong**. If my approach is flawed, I want to know. Open source makes that possible.

**What you get:**

- The Python code (MIT or Apache-2.0)
- Example notebooks
- Documentation

**What you don't get:**

- Pre-trained models (you'll need to download them yourself)
- Datasets (you'll need to provide your own)

---

## FAQ

**Is this finished?**  
No. It's a work in progress. I'm publishing it early to get feedback and collaborators.

**Does it work with MiniMax clones?**  
That's the main use case. We'll find out together.

**Does it work with ElevenLabs?**  
Probably. But the warping will be different. That's what the experiments are for.

**Can I use this in court?**  
You can try. But talk to a lawyer first. This is a technical tool, not a legal opinion.

**Can I contribute?**  
Yes, if you have experience with speaker embeddings, audio forensics, or Python. Contact me.


---

## Contact

**Serious inquiries only.**

**ProtonMail:** `blende_32@protonmail.com`  
**Threema:** `BA46EWMP`


---

## License

**MIT or Apache-2.0** (to be decided)

© 2026 Johannes Wobus — VoicePrint Research

---

_"We show what's possible — and how it's done."_

---

## Latest Updates

- **2026-07-31:** Project concept finalized. Repository created.
- **2026-08-01:** First architecture draft. Initial experiments planned.
- **To be done:** First working prototype.
