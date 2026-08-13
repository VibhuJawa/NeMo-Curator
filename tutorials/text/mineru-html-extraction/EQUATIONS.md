# Equations: MinerU-HTML vs the trafilatura fallback

The [tutorial](README.md) walks through a plain prose page, where the model's
job is easy to follow. This is the case that actually motivates using a model
at all, and it is also the case where the fallback fails hardest.

Everything below is real output from a live run, not written by hand.

## The page

Section 3.4 of [KAN: Kolmogorov–Arnold Networks](https://arxiv.org/html/2404.19756v4),
which sets up a Poisson problem and a PINN loss. The input is that section left
inside the real arXiv page — announcement banner, header, licence line, footer,
funder logos and all. That is ~43 KB of HTML, 36 numbered elements, a
4,752-token prompt.

arXiv's HTML carries every formula twice: as MathML for the browser, and as the
author's original LaTeX in an `alttext` attribute.

```html
<p class="ltx_p">We consider a Poisson equation with zero Dirichlet boundary data.
For <math class="ltx_Math" alttext="\Omega=[-1,1]^{2}" display="inline"><semantics>
  <mrow><mi mathvariant="normal">Ω</mi><mo>=</mo>…</mrow>
  <annotation encoding="application/x-tex">\Omega=[-1,1]^{2}</annotation>
</semantics></math>, consider the PDE</p>
<table id="S3.E2" class="ltx_equationgroup ltx_eqn_table">…
  <math class="ltx_Math" alttext="\displaystyle u_{xx}+u_{yy}" …>…</math>…
</table>
```

## What the model does

The model labels items 1–10 `other` (a bug-report modal, the announcement
banner, arXiv's header and licence line), 11–23 `main`, and 24–36 `other` (the
footer, the HTML-feedback instructions, the funder logos). Rendering what
survives as `mm_md` gives this — elided at `…` and cut short of the closing
paragraph, otherwise verbatim:

```markdown
# KAN: Kolmogorov–Arnold Networks

## 3 KANs are accurate

### 3.4 Solving partial differential equations

We consider a Poisson equation with zero Dirichlet boundary data. For $\Omega=[-1,1]^{2}$ , consider the PDE

<table><tbody><tr><td>$\displaystyle u_{xx}+u_{yy}$</td><td>$\displaystyle=f\quad\text{in}\,\,\Omega\,,$</td><td rowspan="2">(3.2)</td></tr><tr><td>$\displaystyle u$</td><td>$\displaystyle=0\quad\text{on}\,\,\partial\Omega\,.$</td></tr></tbody></table>

We consider the data $f=-\pi^{2}(1+4y^{2})\sin(\pi x)\sin(\pi y^{2})+2\pi\sin(\pi x)\cos(\pi y^{2})$ for which $u=\sin(\pi x)\sin(\pi y^{2})$ is the true solution. We use the framework of physics-informed neural networks (PINNs)[38, 39] to solve this PDE, with the loss function given by

$$\text{loss}_{\text{pde}}=\alpha\text{loss}_{i}+\text{loss}_{b}\coloneqq\alpha\frac{1}{n_{i}}\sum_{i=1}^{n_{i}}|u_{xx}(z_{i})+u_{yy}(z_{i})-f(z_{i})|^{2}+\frac{1}{n_{b}}\sum_{i=1}^{n_{b}}u^{2}\,,$$

where we use $\text{loss}_{i}$ to denote the interior loss, … $\alpha$ is the hyperparameter balancing the effect of the two terms.
```

### …and how that renders

The block above is the literal text the pipeline writes, which is why it is shown
as source. Rendered, it is the actual mathematics:

> We consider a Poisson equation with zero Dirichlet boundary data. For
> $\Omega=[-1,1]^{2}$, consider the PDE
>
> ```math
> \begin{aligned}
> u_{xx}+u_{yy} &= f \quad\text{in}\,\,\Omega\,, \\
> u &= 0 \quad\text{on}\,\,\partial\Omega\,.
> \end{aligned}
> ```
>
> We consider the data
> $f=-\pi^{2}(1+4y^{2})\sin(\pi x)\sin(\pi y^{2})+2\pi\sin(\pi x)\cos(\pi y^{2})$
> for which $u=\sin(\pi x)\sin(\pi y^{2})$ is the true solution. We use the
> framework of physics-informed neural networks (PINNs) to solve this PDE, with
> the loss function given by
>
> ```math
> \text{loss}_{\text{pde}}=\alpha\,\text{loss}_{i}+\text{loss}_{b} :=
> \alpha\frac{1}{n_{i}}\sum_{i=1}^{n_{i}}\left|u_{xx}(z_{i})+u_{yy}(z_{i})-f(z_{i})\right|^{2}
> +\frac{1}{n_{b}}\sum_{i=1}^{n_{b}}u^{2}\,,
> ```

Two deliberate differences from the raw output, neither of them the pipeline's
doing. The PDE arrives as an HTML `<table>` because arXiv marks equation groups
up that way and the renderer preserves it; it is set as an `aligned` block here
because GitHub does not process `$…$` inside HTML tables. And `\coloneqq` is
written `:=`, because that macro comes from
[mathtools](https://docs.mathjax.org/en/v3.2/input/tex/extensions/mathtools.html),
which GitHub's MathJax does not load — paste the raw output into a GitHub file
unchanged and that one symbol fails to render. Worth knowing if you plan to
publish extracted maths straight to GitHub.

## The same page through the fallback

Trimmed at the same point:

```markdown
# KAN: Kolmogorov–Arnold Networks

## 3 KANs are accurate

### 3.4 Solving partial differential equations

We consider a Poisson equation with zero Dirichlet boundary data. For , consider the PDE

(3.2)

We consider the data for which is the true solution. We use the framework of physics-informed neural networks (PINNs) [38, 39] to solve this PDE, with the loss function given by

where we use to denote the interior loss, … is the hyperparameter balancing the effect of the two terms.
```

## What this shows, and what it doesn't

Both extractors found the same article and dropped the same boilerplate — on
this page the fallback's *selection* is fine, and the difference is entirely in
what reaches the renderer. The page has 23 `<math>` elements; the pruned page
still has 23 and trafilatura's has none, so the equation *number* `(3.2)`
survives and no equation does, leaving sentences like "We consider the data for
which is the true solution" that read as complete but say nothing. Note what
this is and isn't: the win is not the model's labelling but the fact that
MinerU-HTML *deletes* nodes from the original DOM, so the MathML and its
`alttext` survive for `mm_md` to turn into LaTeX, whereas trafilatura rebuilds a clean tree
with no place to put a formula — its own Markdown mode drops them too, so this
is not an artefact of how the pipeline calls it.

Back to the [tutorial](README.md).
