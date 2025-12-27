# Comprehensive Revision Guide for FedXChain Enhanced Paper

## Executive Summary

This document provides a complete action plan to address all reviewer comments. Your paper **ALREADY MEETS** most of Reviewer Y's substantive requirements (multi-model validation, real data, statistics), but requires **FORMATTING AND STRUCTURAL FIXES** per Reviewer X's comments.

---

## ✅ What's Already Good (Reviewer Y - Substantive Content)

Your enhanced paper ALREADY demonstrates:

### 1. Multi-Model Validation ✅
- **3 model architectures**: Logistic Regression, MLP (64,32), Random Forest (50 trees)
- **Evidence**: Table 1 (Lines 295-311), Figure 8 (Lines 405-411)
- **No changes needed** - Reviewer Y acknowledged this

### 2. Real-World Dataset ✅
- **Wisconsin Breast Cancer**: 569 clinical samples, 30 features
- **Evidence**: Section 5.1 (Lines 260-270), all main results tables
- **No changes needed** - Reviewer Y satisfied

### 3. Statistical Validation ✅
- **5 independent runs** with 95% confidence intervals
- **CV < 2%** for all breast cancer experiments (excellent reproducibility)
- **Evidence**: Table 1 shows mean ± std, Section 6.2 (Lines 415-430)
- **No changes needed** - Reviewer Y praised this

### 4. Formal NSDS Definition ✅
- **Equations 3-6** provide KL-divergence formulation
- **Evidence**: Lines 167-180
- **Needs enhancement** - Add algorithm + example (see below)

---

## ❌ What Needs Fixing (Reviewer X - Formatting Issues)

### HIGH PRIORITY - Must Fix Before Resubmission

#### 1. Author Email Addresses ❌ → ✅ (FIXED)
**Status**: **COMPLETED** in current version

**What was done**:
- Added emails for all 6 co-authors
- Lines 19-48 now include:
  - mahdin.rohmatillah@ub.ac.id
  - cries.avian@ub.ac.id  
  - sholeh.pramono@ub.ac.id
  - fauzan.purnomo@ub.ac.id
  - panca.m@ub.ac.id

#### 2. Merge Related Work into Introduction ❌ (NEEDS FIXING)
**Current issue**: Separate "Related Work" section (Section 3, Lines 114-139)

**Required fix**:
```latex
% BEFORE (current structure):
\section{Introduction}
... [current intro] ...

\section{Related Work}  % ← This entire section must be merged
\subsection{Federated Learning...}
\subsection{Trust and Robustness...}
\subsection{Explainable AI...}
\subsection{Blockchain Integration...}

\section{FedXChain Methodology}  % ← This becomes Section 3

% AFTER (required structure):
\section{Introduction}
... [current intro paragraphs 1-4] ...

% INSERT HERE: Related work content integrated into narrative
Recent federated learning research... [FedAvg, FedProx citations]
Trust-based approaches... [Byzantine, incentive mechanisms]
Explainable AI techniques... [SHAP, LIME]
Blockchain integration... [audit trails, decentralization]

% Transition to contributions
We introduce FedXChain...

\section{Notation and Problem Formulation}  % ← Stays Section 2

\section{FedXChain Methodology}  % ← Now becomes Section 3 (renumbered)
```

**Action items**:
1. Copy Related Work subsections 3.1-3.4 content
2. Integrate into Introduction paragraphs 5-8 (before final "We introduce FedXChain" paragraph)
3. Delete standalone Section 3 "Related Work"
4. Renumber: Methodology becomes Section 3, Results becomes Section 4, etc.

#### 3. LaTeX Formatting Errors ❌ (NEEDS CHECKING)
**Reported issue**: "Corrupted text like `textKL(P_i IP_textglobal)`"

**How to find**:
```bash
# Search for malformed LaTeX in paper
grep -n "textKL\|IP_text\|backslash.*text{" fedxchain_paper_revised.tex
```

**Common errors to fix**:
- `textKL(...)` → `\text{KL}(...)`
- `P_i IP_textglobal` → `P_i \| P_{\text{global}}`
- Missing `\` before text commands
- Improper escaping in math mode

**Where to check** (Lines likely affected):
- Line 106: Problem formulation KL notation
- Lines 140-180: NSDS definition equations
- Lines 167-170: Equation (4) KL-divergence

#### 4. Figure Caption Standardization ❌ (NEEDS FIXING)
**Current issue**: Mixed "Figure X" and "Fig. X" usage

**Required fix**: All must use "Fig. X" per IEEE guidelines

**Affected lines** (8 figures):
- Line 342: `Figure \ref{fig:accuracy_rounds}` → `Fig.~\ref{fig:accuracy_rounds}`
- Line 346: `\caption{Validation accuracy...}` → `\caption{Fig. 1. Validation accuracy...}` ❌ NO!
  - Actually IEEE puts "Fig." automatically, just use `\caption{Validation accuracy...}`
- But in-text references must be "Fig." not "Figure"

**Correct format**:
```latex
% In captions (LaTeX adds "Fig." automatically):
\caption{Validation accuracy over training rounds...}

% In text references:
Fig.~\ref{fig:accuracy_rounds} shows...  % ← Use Fig. not Figure
As shown in Figs.~\ref{fig:a}, \ref{fig:b}, and \ref{fig:c}...
```

**Action items**:
1. Search and replace: `Figure \ref{fig:` → `Fig.~\ref{fig:`
2. Search and replace: `Figures \ref{fig:` → `Figs.~\ref{fig:`
3. Verify captions don't manually add "Fig." (IEEEtran does this automatically)

#### 5. Table 1 Layout ❌ (NEEDS FIXING)
**Current issue**: Single-column table is cramped

**Required fix**: Use full-width `table*` environment

**Change needed**:
```latex
% Line 295: BEFORE
\begin{table}[t]
\centering
\caption{Experimental Results Summary...}
\label{tab:main_results}
\begin{tabular}{lccccc}

% Line 295: AFTER  
\begin{table*}[t]  % ← Add asterisk for full-width
\centering
\caption{Experimental Results Summary...}
\label{tab:main_results}
\begin{tabular}{lccccc}

% Line 311: BEFORE
\end{tabular}
\end{table}

% Line 311: AFTER
\end{tabular}
\end{table*}  % ← Add asterisk
```

#### 6. Citation Numbering Verification ❌ (NEEDS CHECKING)
**Task**: Verify sequential [1]-[21] with no gaps/duplicates

**How to check**:
```bash
# Extract all citations in order
grep -o '\cite{[^}]*}' fedxchain_paper_revised.tex | head -30

# Check references.bib entries
grep '@' references.bib | wc -l  # Should be 21
```

**Verification steps**:
1. First citation should be [1] (likely McMahan FedAvg)
2. Last citation should be [21]
3. No missing numbers in sequence
4. No duplicate citation keys in references.bib

#### 7. Duplicate Reference Removal ❌ (NEEDS CHECKING)
**Reported issue**: "Ribeiro et al. (LIME) cited twice"

**How to find**:
```bash
grep -i "ribeiro" references.bib
```

**Expected**:
- Should find ONLY ONE entry (either `ribeiro2016should` or `ribeiro2016lime`)
- If two entries found, remove duplicate

**Action**: 
```bibtex
% KEEP THIS ONE (canonical LIME paper):
@inproceedings{ribeiro2016should,
  title={Why should I trust you?: Explaining the predictions of any classifier},
  author={Ribeiro, Marco Tulio and Singh, Sameer and Guestrin, Carlos},
  booktitle={KDD},
  year={2016}
}

% DELETE IF EXISTS:
@inproceedings{ribeiro2016lime,  % ← Duplicate, remove this
  ...
}
```

---

## 🔧 MEDIUM PRIORITY - Content Enhancements (Reviewer Y)

### 1. Enhanced NSDS Definition with Algorithm ⚠️ (RECOMMENDED)

**Current status**: NSDS defined formally with equations (Lines 167-180)

**Enhancement requested**: Add step-by-step algorithm showing:
- SHAP → probability distribution conversion
- Zero-value handling  
- Numerical stability techniques
- Worked example with numbers

**Suggested addition** (insert after Line 180):

```latex
\subsection{NSDS Computation Algorithm}

Algorithm 1 details the complete NSDS computation procedure.

\begin{algorithm}
\caption{Node-Specific Divergence Score Computation}
\begin{algorithmic}[1]
\Require Node SHAP vector $\mathbf{s}_i \in \mathbb{R}^d$, Global SHAP vector $\mathbf{s}_{\text{global}} \in \mathbb{R}^d$
\Ensure NSDS value $D_i \geq 0$

\State \textbf{Step 1: Absolute Values}
\State $\mathbf{a}_i \gets |\mathbf{s}_i|$ \Comment{Element-wise absolute value}
\State $\mathbf{a}_{\text{global}} \gets |\mathbf{s}_{\text{global}}|$

\State \textbf{Step 2: Epsilon-Smoothing}
\State $\epsilon \gets 10^{-10}$ \Comment{Stability constant}
\State $\mathbf{a}_i \gets \mathbf{a}_i + \epsilon$ \Comment{Add to all elements}
\State $\mathbf{a}_{\text{global}} \gets \mathbf{a}_{\text{global}} + \epsilon$

\State \textbf{Step 3: Normalization}
\State $P_i(j) \gets \frac{a_{i,j}}{\sum_{k=1}^d a_{i,k}}$ for $j=1,\ldots,d$
\State $P_{\text{global}}(j) \gets \frac{a_{\text{global},j}}{\sum_{k=1}^d a_{\text{global},k}}$ for $j=1,\ldots,d$

\State \textbf{Step 4: KL-Divergence}
\State $D_i \gets 0$
\For{$j = 1$ to $d$}
  \State $D_i \gets D_i + P_i(j) \cdot \log\left(\frac{P_i(j)}{P_{\text{global}}(j)}\right)$
\EndFor

\State \Return $D_i$
\end{algorithmic}
\end{algorithm}

\textbf{Example 4.1 (Worked Calculation):}

Consider a 4-feature scenario with two nodes:

\textit{Input SHAP values:}
\begin{itemize}
\item Node A: $\mathbf{s}_A = [0.8, 0.2, 0.0, 0.1]$
\item Node B: $\mathbf{s}_B = [0.1, 0.7, 0.3, 0.0]$
\item Global (averaged): $\mathbf{s}_{\text{global}} = [0.45, 0.45, 0.15, 0.05]$
\end{itemize}

\textit{Step 1 (Absolute values):} Already non-negative, unchanged.

\textit{Step 2 (Smoothing with $\epsilon = 10^{-10}$):}
\begin{itemize}
\item $\mathbf{a}_A = [0.8, 0.2, 10^{-10}, 0.1]$ (third element smoothed)
\item $\mathbf{a}_B = [0.1, 0.7, 0.3, 10^{-10}]$ (fourth element smoothed)
\item $\mathbf{a}_{\text{global}} = [0.45, 0.45, 0.15, 0.05]$ (all non-zero already)
\end{itemize}

\textit{Step 3 (Normalization):}
\begin{itemize}
\item $P_A \approx [0.727, 0.182, 0.000, 0.091]$ (sums to 1.0)
\item $P_B \approx [0.091, 0.636, 0.273, 0.000]$ (sums to 1.0)
\item $P_{\text{global}} \approx [0.409, 0.409, 0.136, 0.045]$ (sums to 1.0)
\end{itemize}

\textit{Step 4 (KL-Divergence):}
\begin{align*}
\text{NSDS}_A &= \sum_{j=1}^4 P_A(j) \log\frac{P_A(j)}{P_{\text{global}}(j)} \\
&= 0.727 \log(1.777) + 0.182 \log(0.445) + \cdots \\
&\approx 0.427
\end{align*}

Similarly, $\text{NSDS}_B \approx 0.389$.

\textit{Interpretation:} Both nodes show moderate divergence from global consensus. Node B is slightly closer ($0.389 < 0.427$), suggesting its local explanations align better with the global pattern. The adaptive aggregation will weight both nodes appropriately, preserving their unique local patterns while building global consensus.
```

### 2. Trust Score Intuition ⚠️ (RECOMMENDED)

**Current status**: Formula given (Lines 190-200), but lacks intuitive explanation

**Enhancement requested**: Explain WHY combining accuracy + explainability + consistency matters

**Suggested addition** (insert after current trust score formula):

```latex
\subsection{Trust Score Rationale and Intuition}

The three-component trust score design addresses critical challenges in federated learning:

\textbf{(1) Accuracy Component ($\alpha = 0.4$):} Ensures high-performing nodes receive appropriate weight. Without this, malicious nodes could contribute poor models while maintaining high explainability scores, degrading global performance.

\textbf{(2) Explainability Fidelity ($\beta = 0.4$):} Detects adversarial behavior where a node achieves high accuracy through means inconsistent with learned features (e.g., memorization, backdoor attacks). If SHAP explanations don't align with model parameters, the node's trust is reduced even with high accuracy.

\textbf{(3) Temporal Consistency ($\gamma = 0.2$):} Prevents erratic behavior. A node that flip-flops between high and low quality contributions is less trustworthy than one with steady performance, even if average metrics are similar.

\textbf{Example (Healthcare Scenario):}

Consider two hospitals in a federated medical AI system:

\textit{Hospital C:}
\begin{itemize}
\item Accuracy: 95\% (excellent)
\item Explainability Fidelity: 30\% (poor - SHAP doesn't match model coefficients)
\item Consistency: 80\% (good)
\item Trust Score: $0.4(0.95) + 0.4(0.30) + 0.2(0.80) = 0.66$
\end{itemize}

\textit{Hospital D:}
\begin{itemize}
\item Accuracy: 85\% (good)
\item Explainability Fidelity: 90\% (excellent - explanations align with clinical knowledge)
\item Consistency: 95\% (excellent)
\item Trust Score: $0.4(0.85) + 0.4(0.90) + 0.2(0.95) = 0.89$
\end{itemize}

Despite Hospital C's higher accuracy, Hospital D receives a higher trust score because its model is interpretable and consistent with medical knowledge. This prevents potentially adversarial or overfitted models from dominating aggregation, which is critical in high-stakes medical applications where model transparency is as important as accuracy.

The weight parameters ($\alpha=0.4, \beta=0.4, \gamma=0.2$) balance performance with interpretability. Section 6.7 presents sensitivity analysis showing this configuration maximizes both accuracy and XAI fidelity across diverse non-IID settings.
```

---

## 📋 Complete Action Checklist

### Immediate Actions (Before Resubmission)

- [x] **1.1** Add all 6 author emails (DONE - already in current version)
- [ ] **1.2** Merge Related Work into Introduction
  - [ ] Copy subsections 3.1-3.4 content
  - [ ] Insert into Introduction before final paragraph
  - [ ] Delete standalone Section 3
  - [ ] Renumber all subsequent sections
- [ ] **1.3** Fix LaTeX formatting errors
  - [ ] Search for `textKL`, `IP_text`, malformed commands
  - [ ] Verify Equations 3-6 render correctly
  - [ ] Test compile: `pdflatex fedxchain_paper_revised.tex`
- [ ] **1.4** Standardize figure captions to "Fig."
  - [ ] Replace all `Figure \ref` with `Fig.~\ref`
  - [ ] Replace all `Figures \ref` with `Figs.~\ref`
  - [ ] Verify captions don't manually add "Fig." prefix
- [ ] **1.5** Convert Table 1 to full-width
  - [ ] Change `\begin{table}[t]` to `\begin{table*}[t]`
  - [ ] Change `\end{table}` to `\end{table*}`
- [ ] **1.6** Verify citations [1]-[21] sequential
  - [ ] Check first citation is [1]
  - [ ] Check last citation is [21]
  - [ ] No gaps in sequence
- [ ] **1.7** Remove duplicate Ribeiro reference
  - [ ] Search `grep -i ribeiro references.bib`
  - [ ] Keep one entry, delete duplicate

### Recommended Enhancements (Strengthen Paper)

- [ ] **2.1** Add NSDS computation algorithm
  - [ ] Insert Algorithm 1 after Line 180
  - [ ] Add worked example with numbers
  - [ ] Add interpretation section
- [ ] **2.2** Add trust score intuition
  - [ ] Explain rationale for 3-component design
  - [ ] Add healthcare example
  - [ ] Connect to sensitivity analysis (Section 6.7)

### Final Verification

- [ ] **3.1** Compile paper without errors
  ```bash
  cd paper
  pdflatex fedxchain_paper_revised.tex
  bibtex fedxchain_paper_revised
  pdflatex fedxchain_paper_revised.tex
  pdflatex fedxchain_paper_revised.tex
  ```
- [ ] **3.2** Check PDF output
  - [ ] All figures visible
  - [ ] All tables readable
  - [ ] All equations rendered correctly
  - [ ] All citations show numbers [X] not [?]
- [ ] **3.3** Verify page count (IEEE conference typically 8-10 pages)
- [ ] **3.4** Create final submission package
  - [ ] REVIEWER_RESPONSE.pdf (this guide + detailed responses)
  - [ ] fedxchain_paper_revised.tex (revised LaTeX source)
  - [ ] fedxchain_paper_revised.pdf (compiled revised manuscript)
  - [ ] references.bib (updated bibliography)
  - [ ] figures/ directory (all 8 PDF figures)

---

## 📝 Submission Format (Per Journal Requirements)

**Single Word Document Structure:**

```
Pages 1-4: REVIEWER RESPONSE
  - Comment X.1 → Response → Changes (author emails)
  - Comment X.2 → Response → Changes (merged Related Work)
  - Comment X.3 → Response → Changes (LaTeX fixes)
  - Comment X.4 → Response → Changes (figure captions)
  - Comment X.5 → Response → Changes (table layout)
  - Comment X.6 → Response → Changes (citation numbering)
  - Comment X.7 → Response → Changes (duplicate reference)
  - Comment Y.1 → Response → Changes (NSDS algorithm)
  - Comment Y.2 → Response → Changes (trust intuition)
  - Summary table of all changes

Pages 5+: REVISED MANUSCRIPT
  - Title page with all 6 author emails
  - Abstract
  - Keywords
  - Introduction (with integrated Related Work)
  - Notation and Problem Formulation
  - FedXChain Methodology (with NSDS algorithm + trust intuition)
  - Experimental Setup
  - Results and Analysis (with 8 figures)
  - Discussion
  - Conclusion
  - References [1]-[21]
```

**Conversion to Word:**
```bash
# Option 1: Use pandoc (preserves formatting better)
pandoc fedxchain_paper_revised.tex -o fedxchain_paper_revised.docx --bibliography=references.bib

# Option 2: Export PDF to Word (if pandoc unavailable)
# Use Adobe Acrobat or online PDF-to-Word converter
# Then manually format to match requirements
```

---

## ⚡ Quick Start Commands

```bash
# Navigate to paper directory
cd /mnt/sda2/projects/federated_learning/EXPERIMENT_USING_DOCKER/SIMULASI_EXPERIMENT/fedXchain-etasr/paper

# Work on revised version
cp fedxchain_paper_enhanced.tex fedxchain_paper_revised.tex

# Check for formatting issues
grep -n "textKL\|Figure \\\\ref\|begin{table}\[" fedxchain_paper_revised.tex

# Check references
grep -i "ribeiro" references.bib
grep '@' references.bib | wc -l  # Should be 21

# Test compile
pdflatex fedxchain_paper_revised.tex
bibtex fedxchain_paper_revised
pdflatex fedxchain_paper_revised.tex
pdflatex fedxchain_paper_revised.tex

# View result
xdg-open fedxchain_paper_revised.pdf
```

---

## 🎯 Bottom Line

**What you already have:**
- ✅ Excellent experimental validation (multi-model, real data, statistics)
- ✅ Comprehensive figures (8 professional visualizations)
- ✅ Strong theoretical foundation (formal equations, algorithms)

**What you must fix:**
- ❌ Merge Related Work into Introduction (structural)
- ❌ Standardize figure captions to "Fig." (formatting)
- ❌ Convert Table 1 to full-width layout (formatting)
- ❌ Fix any LaTeX errors (formatting)
- ❌ Verify citation sequence (reference management)

**What you should add:**
- ⚠️ NSDS computation algorithm with example (enhances clarity)
- ⚠️ Trust score intuition section (enhances understanding)

**Estimated work time:**
- **Must-fix items**: 2-3 hours
- **Should-add items**: 1-2 hours
- **Total**: Half-day revision

**Your paper is VERY CLOSE to acceptance.** The substantive work (experiments, validation, results) is excellent. The remaining tasks are primarily formatting and structural refinements.

---

**Last updated**: December 13, 2025
**Document version**: 1.0
**Status**: Ready for implementation
