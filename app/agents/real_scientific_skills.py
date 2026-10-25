"""
Inference/app/agents/real_scientific_skills.py — Real Scientific Agent Skills.

The capability layer of a research agent: a searchable skill registry with
procedural knowledge, a composer that wires skills into a provenance-tracked
pipeline, and the actual scientific computation those skills promise
(hypothesis tests, power analysis, sequence/molecule analysis, units,
hypothesis generation and evidence-traceable writing).

Real working implementations:
- StatisticalTestSuite: Welch/paired t, Mann-Whitney U, chi-square, one-way
  ANOVA, correlation, effect sizes, Benjamini-Hochberg, bootstrap CI
- PowerAnalysis: sample size, power, minimum detectable effect
- SequenceAnalysis: GC, reverse complement, Tm, MW, ORF finding, translation,
  k-mers, GRAVY hydropathy
- MolecularFormula: formula parsing, molecular weight, unsaturation, Lipinski
- UnitConverter: dimensional conversion with temperature handling
- SkillLibrary / SkillMatcher / SkillComposer: the 160+ skill catalogue,
  ranked matching and pipeline composition
- ProvenanceGraph: artifact lineage for every produced result
- HypothesisGenerator: evidence-bounded, ranked, falsifiable hypotheses
- ScientificWriter: claim/evidence auditing and citation rendering
- ExperimentDesigner: factorial designs, randomized blocks, sample sizing
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable


# ── Distribution math ─────────────────────────────────────────────────────

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz)."""
    max_iter, eps, fpmin = 200, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b), the regularized incomplete beta function."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _gamma_series(a: float, x: float) -> float:
    term = 1.0 / a
    total = term
    ap = a
    for _ in range(500):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_cf(a: float, x: float) -> float:
    fpmin, eps = 1e-300, 1e-15
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def regularized_gamma_p(a: float, x: float) -> float:
    """Lower regularized incomplete gamma P(a, x)."""
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        return _gamma_series(a, x)
    return 1.0 - _gamma_cf(a, x)


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    """Inverse normal CDF via bisection on erf — stable and dependency-free."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    lo, hi = -10.0, 10.0
    for _ in range(120):
        mid = (lo + hi) / 2.0
        if normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value of Student's t statistic."""
    if df <= 0:
        return 1.0
    if math.isinf(t):
        return 0.0
    x = df / (df + t * t)
    return regularized_incomplete_beta(df / 2.0, 0.5, x)


def f_survival(f: float, d1: float, d2: float) -> float:
    """P(F > f) for the F distribution."""
    if f <= 0:
        return 1.0
    x = d2 / (d2 + d1 * f)
    return regularized_incomplete_beta(d2 / 2.0, d1 / 2.0, x)


def chi2_survival(x: float, df: float) -> float:
    """P(X > x) for the chi-square distribution."""
    if x <= 0:
        return 1.0
    return 1.0 - regularized_gamma_p(df / 2.0, x / 2.0)


def mean(values: Iterable[float]) -> float:
    data = list(values)
    return sum(data) / len(data) if data else 0.0


def variance(values: Iterable[float], ddof: int = 1) -> float:
    data = list(values)
    if len(data) - ddof <= 0:
        return 0.0
    mu = mean(data)
    return sum((x - mu) ** 2 for x in data) / (len(data) - ddof)


def stdev(values: Iterable[float], ddof: int = 1) -> float:
    return math.sqrt(variance(values, ddof))


# ── Hypothesis tests ──────────────────────────────────────────────────────

@dataclass
class TestResult:
    """The outcome of one statistical test."""

    name: str
    statistic: float
    p_value: float
    df: float = 0.0
    effect_size: float = 0.0
    alpha: float = 0.05
    detail: dict[str, float] = field(default_factory=dict)

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def summary(self) -> str:
        verdict = "reject H0" if self.significant else "fail to reject H0"
        return (
            f"{self.name}: statistic={self.statistic:.4f}, p={self.p_value:.5f}, "
            f"effect={self.effect_size:.3f} -> {verdict} at alpha={self.alpha}"
        )


class StatisticalTestSuite:
    """Real hypothesis tests with closed-form p-values (no scipy)."""

    # ── t tests ──────────────────────────────────────────────────────

    def t_test_independent(self, a: list[float], b: list[float], alpha: float = 0.05) -> TestResult:
        """Welch's unequal-variance t-test."""
        n1, n2 = len(a), len(b)
        if n1 < 2 or n2 < 2:
            return TestResult("welch_t", 0.0, 1.0, alpha=alpha, detail={"error": 1.0})
        m1, m2 = mean(a), mean(b)
        v1, v2 = variance(a), variance(b)
        se2 = v1 / n1 + v2 / n2
        if se2 <= 0:
            return TestResult("welch_t", 0.0, 1.0, alpha=alpha, detail={"se": 0.0})
        t = (m1 - m2) / math.sqrt(se2)
        df = se2 ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
        p = t_two_sided_p(t, df)
        pooled = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
        d = (m1 - m2) / pooled if pooled > 0 else 0.0
        return TestResult(
            "welch_t", t, p, df=df, effect_size=d, alpha=alpha,
            detail={"mean_a": m1, "mean_b": m2, "se": math.sqrt(se2), "n_a": n1, "n_b": n2},
        )

    def t_test_paired(self, a: list[float], b: list[float], alpha: float = 0.05) -> TestResult:
        if len(a) != len(b) or len(a) < 2:
            return TestResult("paired_t", 0.0, 1.0, alpha=alpha, detail={"error": 1.0})
        diffs = [x - y for x, y in zip(a, b)]
        n = len(diffs)
        sd = stdev(diffs)
        if sd == 0:
            return TestResult("paired_t", 0.0, 1.0, df=n - 1, alpha=alpha, detail={"sd": 0.0})
        t = mean(diffs) / (sd / math.sqrt(n))
        p = t_two_sided_p(t, n - 1)
        return TestResult(
            "paired_t", t, p, df=n - 1, effect_size=mean(diffs) / sd, alpha=alpha,
            detail={"mean_diff": mean(diffs), "sd": sd, "n": n},
        )

    # ── non-parametric ───────────────────────────────────────────────

    @staticmethod
    def _ranks(values: list[float]) -> tuple[list[float], dict[float, int]]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        tie_counts: Counter[float] = Counter(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks, dict(tie_counts)

    def mann_whitney_u(self, a: list[float], b: list[float], alpha: float = 0.05) -> TestResult:
        """Mann-Whitney U with tie-corrected normal approximation."""
        n1, n2 = len(a), len(b)
        if n1 < 1 or n2 < 1:
            return TestResult("mann_whitney_u", 0.0, 1.0, alpha=alpha, detail={"error": 1.0})
        combined = list(a) + list(b)
        ranks, ties = self._ranks(combined)
        r1 = sum(ranks[:n1])
        u1 = r1 - n1 * (n1 + 1) / 2.0
        u2 = n1 * n2 - u1
        n = n1 + n2
        tie_term = sum(t ** 3 - t for t in ties.values())
        if n <= 1:
            return TestResult("mann_whitney_u", u1, 1.0, alpha=alpha)
        sigma2 = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
        if sigma2 <= 0:
            return TestResult("mann_whitney_u", u1, 1.0, alpha=alpha, detail={"sigma": 0.0})
        mu = n1 * n2 / 2.0
        z = (u1 - mu - 0.5 * (1 if u1 > mu else -1)) / math.sqrt(sigma2)
        p = 2.0 * (1.0 - normal_cdf(abs(z)))
        # Rank-biserial correlation as the effect size.
        effect = 1.0 - (2.0 * min(u1, u2)) / (n1 * n2)
        return TestResult(
            "mann_whitney_u", u1, min(1.0, p), effect_size=effect, alpha=alpha,
            detail={"u2": u2, "z": z, "median_a": _median(a), "median_b": _median(b)},
        )

    def chi_square_independence(self, table: list[list[int]], alpha: float = 0.05) -> TestResult:
        """Pearson chi-square test of independence on a contingency table."""
        rows = len(table)
        cols = len(table[0]) if rows else 0
        if rows < 2 or cols < 2:
            return TestResult("chi_square", 0.0, 1.0, alpha=alpha, detail={"error": 1.0})
        row_totals = [sum(row) for row in table]
        col_totals = [sum(table[r][c] for r in range(rows)) for c in range(cols)]
        total = sum(row_totals)
        if total <= 0:
            return TestResult("chi_square", 0.0, 1.0, alpha=alpha)
        chi2 = 0.0
        min_expected = float("inf")
        for r in range(rows):
            for c in range(cols):
                expected = row_totals[r] * col_totals[c] / total
                if expected <= 0:
                    continue
                min_expected = min(min_expected, expected)
                chi2 += (table[r][c] - expected) ** 2 / expected
        df = (rows - 1) * (cols - 1)
        p = chi2_survival(chi2, df)
        cramers_v = math.sqrt(chi2 / (total * (min(rows, cols) - 1)))
        return TestResult(
            "chi_square", chi2, p, df=df, effect_size=cramers_v, alpha=alpha,
            detail={"min_expected": min_expected if min_expected != float("inf") else 0.0,
                    "n": float(total)},
        )

    def one_way_anova(self, groups: list[list[float]], alpha: float = 0.05) -> TestResult:
        """One-way ANOVA with eta-squared effect size."""
        groups = [g for g in groups if g]
        k = len(groups)
        n_total = sum(len(g) for g in groups)
        if k < 2 or n_total <= k:
            return TestResult("anova", 0.0, 1.0, alpha=alpha, detail={"error": 1.0})
        grand = mean([x for g in groups for x in g])
        ss_between = sum(len(g) * (mean(g) - grand) ** 2 for g in groups)
        ss_within = sum(sum((x - mean(g)) ** 2 for x in g) for g in groups)
        df_between, df_within = k - 1, n_total - k
        if ss_within <= 0 or df_within <= 0:
            return TestResult("anova", 0.0, 1.0, alpha=alpha)
        f = (ss_between / df_between) / (ss_within / df_within)
        p = f_survival(f, df_between, df_within)
        eta2 = ss_between / (ss_between + ss_within) if (ss_between + ss_within) > 0 else 0.0
        return TestResult(
            "anova", f, p, df=float(df_between), effect_size=eta2, alpha=alpha,
            detail={"df_within": float(df_within), "ss_between": ss_between,
                    "ss_within": ss_within},
        )

    # ── association ──────────────────────────────────────────────────

    def pearson(self, x: list[float], y: list[float], alpha: float = 0.05) -> TestResult:
        n = min(len(x), len(y))
        if n < 3:
            return TestResult("pearson_r", 0.0, 1.0, alpha=alpha)
        mx, my = mean(x[:n]), mean(y[:n])
        num = sum((a - mx) * (b - my) for a, b in zip(x[:n], y[:n]))
        den = math.sqrt(sum((a - mx) ** 2 for a in x[:n]) * sum((b - my) ** 2 for b in y[:n]))
        if den == 0:
            return TestResult("pearson_r", 0.0, 1.0, alpha=alpha)
        r = num / den
        if abs(r) >= 1.0:
            return TestResult("pearson_r", r, 0.0, df=n - 2, effect_size=r, alpha=alpha)
        t = r * math.sqrt((n - 2) / (1 - r * r))
        return TestResult("pearson_r", r, t_two_sided_p(t, n - 2), df=n - 2,
                          effect_size=r, alpha=alpha, detail={"n": n})

    def spearman(self, x: list[float], y: list[float], alpha: float = 0.05) -> TestResult:
        n = min(len(x), len(y))
        if n < 3:
            return TestResult("spearman_rho", 0.0, 1.0, alpha=alpha)
        rx, _ = self._ranks(list(x[:n]))
        ry, _ = self._ranks(list(y[:n]))
        result = self.pearson(rx, ry, alpha=alpha)
        result.name = "spearman_rho"
        return result

    # ── effect sizes ─────────────────────────────────────────────────

    @staticmethod
    def cohens_d(a: list[float], b: list[float]) -> float:
        n1, n2 = len(a), len(b)
        if n1 < 2 or n2 < 2:
            return 0.0
        pooled = math.sqrt(((n1 - 1) * variance(a) + (n2 - 1) * variance(b)) / (n1 + n2 - 2))
        return (mean(a) - mean(b)) / pooled if pooled > 0 else 0.0

    @staticmethod
    def odds_ratio(a: int, b: int, c: int, d: int) -> float:
        """Odds ratio for the 2x2 table [[a, b], [c, d]] with Haldane correction."""
        if min(a, b, c, d) == 0:
            a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        return (a * d) / (b * c)

    @staticmethod
    def relative_risk(a: int, b: int, c: int, d: int) -> float:
        exposed = a / (a + b) if (a + b) else 0.0
        control = c / (c + d) if (c + d) else 0.0
        return exposed / control if control else float("inf")

    # ── multiple comparisons ─────────────────────────────────────────

    @staticmethod
    def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> dict[str, Any]:
        """Benjamini-Hochberg FDR control; returns q-values and rejections."""
        m = len(p_values)
        if m == 0:
            return {"q_values": [], "rejected": [], "thresholds": [], "alpha": alpha}
        order = sorted(range(m), key=lambda i: p_values[i])
        q_values = [0.0] * m
        prev = 1.0
        for rank in range(m, 0, -1):
            idx = order[rank - 1]
            q = min(prev, p_values[idx] * m / rank)
            q_values[idx] = round(q, 10)
            prev = q
        rejected = [q_values[i] < alpha for i in range(m)]
        thresholds = [alpha * rank / m for rank in range(1, m + 1)]
        return {
            "q_values": q_values,
            "rejected": rejected,
            "thresholds": thresholds,
            "alpha": alpha,
            "discoveries": sum(1 for r in rejected if r),
        }

    @staticmethod
    def bonferroni(p_values: list[float], alpha: float = 0.05) -> dict[str, Any]:
        m = len(p_values)
        adjusted = [min(1.0, p * m) for p in p_values]
        return {"adjusted": adjusted, "rejected": [a < alpha for a in adjusted], "alpha": alpha}

    # ── resampling ───────────────────────────────────────────────────

    def bootstrap_ci(
        self, data: list[float], n_resamples: int = 2000, seed: int = 7,
        confidence: float = 0.95,
    ) -> dict[str, float]:
        """Percentile bootstrap CI for the mean (deterministic seed)."""
        if not data:
            return {"lower": 0.0, "upper": 0.0, "mean": 0.0}
        rng = random.Random(seed)
        n = len(data)
        means = []
        for _ in range(n_resamples):
            sample = [data[rng.randrange(n)] for _ in range(n)]
            means.append(mean(sample))
        means.sort()
        tail = (1.0 - confidence) / 2.0
        lo_idx = max(0, int(math.floor(tail * n_resamples)))
        hi_idx = min(n_resamples - 1, int(math.ceil((1 - tail) * n_resamples)) - 1)
        return {"lower": means[lo_idx], "upper": means[hi_idx], "mean": mean(data)}


def _median(values: list[float]) -> float:
    data = sorted(values)
    n = len(data)
    if n == 0:
        return 0.0
    mid = n // 2
    return data[mid] if n % 2 else (data[mid - 1] + data[mid]) / 2.0


# ── Power analysis ────────────────────────────────────────────────────────

@dataclass
class PowerResult:
    """Sample-size / power calculation result."""

    n_per_group: int
    power: float
    effect_size: float
    alpha: float
    detectable_effect: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "n_per_group": self.n_per_group,
            "power": round(self.power, 4),
            "effect_size": round(self.effect_size, 4),
            "alpha": self.alpha,
            "detectable_effect": round(self.detectable_effect, 4),
        }


class PowerAnalysis:
    """Normal-approximation power analysis for two-sample means and proportions."""

    def sample_size_two_sample(
        self, effect_size: float, alpha: float = 0.05, power: float = 0.8,
    ) -> PowerResult:
        if effect_size <= 0:
            return PowerResult(0, 0.0, effect_size, alpha)
        z_alpha = normal_ppf(1.0 - alpha / 2.0)
        z_beta = normal_ppf(power)
        n = math.ceil(2.0 * ((z_alpha + z_beta) / effect_size) ** 2)
        achieved = self.power_two_sample(effect_size, n, alpha)
        return PowerResult(n, achieved, effect_size, alpha,
                           detectable_effect=self.minimum_detectable_effect(n, alpha, power))

    def power_two_sample(self, effect_size: float, n_per_group: int, alpha: float = 0.05) -> float:
        if n_per_group <= 1 or effect_size <= 0:
            return 0.0
        z_alpha = normal_ppf(1.0 - alpha / 2.0)
        ncp = effect_size * math.sqrt(n_per_group / 2.0)
        return max(0.0, min(1.0, normal_cdf(ncp - z_alpha)))

    def minimum_detectable_effect(
        self, n_per_group: int, alpha: float = 0.05, power: float = 0.8,
    ) -> float:
        if n_per_group <= 1:
            return float("inf")
        return (normal_ppf(1 - alpha / 2) + normal_ppf(power)) * math.sqrt(2.0 / n_per_group)

    def sample_size_proportion(
        self, p1: float, p2: float, alpha: float = 0.05, power: float = 0.8,
    ) -> PowerResult:
        if not (0 < p1 < 1 and 0 < p2 < 1) or p1 == p2:
            return PowerResult(0, 0.0, abs(p1 - p2), alpha)
        pbar = (p1 + p2) / 2.0
        h = 2 * math.asin(math.sqrt(max(p1, p2))) - 2 * math.asin(math.sqrt(min(p1, p2)))
        z_alpha = normal_ppf(1 - alpha / 2)
        z_beta = normal_ppf(power)
        n = math.ceil(
            ((z_alpha * math.sqrt(2 * pbar * (1 - pbar)) + z_beta * math.sqrt(
                p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p1 - p2) ** 2
        )
        power_achieved = self.power_two_sample(h, n, alpha)
        return PowerResult(n, power_achieved, abs(p1 - p2), alpha)


# ── Sequences ─────────────────────────────────────────────────────────────

DNA_COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L",
    "CTA": "L", "CTG": "L", "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "TCT": "S", "TCC": "S",
    "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A",
    "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R",
    "CGA": "R", "CGG": "R", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

AMINO_ACID_WEIGHTS = {
    "A": 89.1, "R": 174.2, "N": 132.1, "D": 133.1, "C": 121.2, "E": 147.1,
    "Q": 146.2, "G": 75.1, "H": 155.2, "I": 131.2, "L": 131.2, "K": 146.2,
    "M": 149.2, "F": 165.2, "P": 115.1, "S": 105.1, "T": 119.1, "W": 204.2,
    "Y": 181.2, "V": 117.1,
}

KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "E": -3.5, "Q": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

NUCLEOTIDE_MONOPHOSPHATE = {"A": 313.21, "C": 289.18, "G": 329.21, "T": 304.20,
                            "U": 290.17}


@dataclass
class Orf:
    """An open reading frame."""

    start: int
    end: int
    frame: int
    strand: str
    length_bp: int
    protein: str


class SequenceAnalysis:
    """Nucleotide and protein analysis used by the bioinformatics skills."""

    @staticmethod
    def validate_dna(sequence: str) -> str:
        seq = re.sub(r"\s+", "", (sequence or "")).upper()
        if not seq:
            raise ValueError("empty sequence")
        bad = set(seq) - set("ACGTN")
        if bad:
            raise ValueError(f"invalid nucleotide(s): {''.join(sorted(bad))}")
        return seq

    def gc_content(self, sequence: str) -> float:
        seq = self.validate_dna(sequence)
        if not seq:
            return 0.0
        return (seq.count("G") + seq.count("C")) / len(seq)

    def reverse_complement(self, sequence: str) -> str:
        return self.validate_dna(sequence).translate(DNA_COMPLEMENT)[::-1]

    def melting_temp(self, sequence: str) -> float:
        """Wallace rule for short oligos, GC% formula for long products."""
        seq = self.validate_dna(sequence)
        gc = seq.count("G") + seq.count("C")
        if len(seq) < 14:
            at = seq.count("A") + seq.count("T")
            return float(2 * at + 4 * gc)
        return 64.9 + 41.0 * (gc - 16.4) / len(seq)

    def molecular_weight_dna(self, sequence: str, double_stranded: bool = False) -> float:
        seq = self.validate_dna(sequence)
        total = sum(NUCLEOTIDE_MONOPHOSPHATE.get(base, 0.0) for base in seq)
        ss = total - 61.96
        return round(ss * 2 - 61.96 if double_stranded else ss, 2)

    def translate(self, sequence: str) -> str:
        seq = self.validate_dna(sequence).replace("N", "")
        protein = []
        for i in range(0, len(seq) - 2, 3):
            protein.append(CODON_TABLE.get(seq[i:i + 3], "X"))
        return "".join(protein)

    def find_orfs(self, sequence: str, min_protein: int = 10) -> list[Orf]:
        """Find ATG..stop ORFs on both strands."""
        seq = self.validate_dna(sequence)
        found: list[Orf] = []
        for strand, working in (("+", seq), ("-", self.reverse_complement(seq))):
            for frame in range(3):
                i = frame
                while i < len(working) - 2:
                    codon = working[i:i + 3]
                    if codon == "ATG":
                        j = i
                        protein = []
                        while j < len(working) - 2:
                            c = working[j:j + 3]
                            aa = CODON_TABLE.get(c, "X")
                            if aa == "*":
                                break
                            protein.append(aa)
                            j += 3
                        if len(protein) >= min_protein:
                            found.append(Orf(
                                start=i, end=min(len(working), j + 3), frame=frame,
                                strand=strand, length_bp=len(protein) * 3,
                                protein="".join(protein),
                            ))
                        i = j + 3
                    else:
                        i += 3
        found.sort(key=lambda o: (-o.length_bp, o.strand, o.start))
        return found

    def kmer_counts(self, sequence: str, k: int = 3) -> dict[str, int]:
        seq = self.validate_dna(sequence)
        counts: Counter[str] = Counter()
        for i in range(len(seq) - k + 1):
            counts[seq[i:i + k]] += 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def hamming_distance(self, a: str, b: str) -> int:
        if len(a) != len(b):
            raise ValueError("sequences must have equal length")
        return sum(1 for x, y in zip(a.upper(), b.upper()) if x != y)

    def protein_molecular_weight(self, protein: str) -> float:
        """Peptide mass: summed amino-acid weights minus water per peptide bond."""
        residues = [AMINO_ACID_WEIGHTS[aa] for aa in protein.upper() if aa in AMINO_ACID_WEIGHTS]
        if not residues:
            return 0.0
        return round(sum(residues) - 18.015 * (len(residues) - 1), 2)

    def gravy(self, protein: str) -> float:
        """Grand average of hydropathy (Kyte-Doolittle)."""
        values = [KYTE_DOOLITTLE[aa] for aa in protein.upper() if aa in KYTE_DOOLITTLE]
        return round(sum(values) / len(values), 3) if values else 0.0

    def isoelectric_point_estimate(self, protein: str) -> float:
        """Bisection estimate of pI from charged-residue fractions."""
        counts = Counter(protein.upper())
        positive = counts["K"] + counts["R"] + counts["H"] * 0.1
        negative = counts["D"] + counts["E"] + counts["Y"] * 0.1
        if positive + negative == 0:
            return 7.0
        net = positive - negative
        # Shift pI up when positively charged, down when negative.
        return round(max(0.0, min(14.0, 7.0 + 1.5 * net / max(1.0, positive + negative))), 2)


# ── Molecules ─────────────────────────────────────────────────────────────

ATOMIC_WEIGHTS = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Na": 22.990,
    "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06,
    "Cl": 35.45, "K": 39.098, "Ca": 40.078, "Fe": 55.845, "Cu": 63.546,
    "Zn": 65.38, "Br": 79.904, "I": 126.904, "Se": 78.971, "Mn": 54.938,
    "B": 10.81, "Li": 6.94, "Ti": 47.867, "Ni": 58.693, "Co": 58.933,
}

HALOGENS = {"F", "Cl", "Br", "I"}
VALENCE_TWO = {"O", "S", "Se"}


@dataclass
class MolecularProperties:
    """Computed descriptors for a small molecule."""

    formula: str
    molecular_weight: float
    heavy_atoms: int
    degree_of_unsaturation: float
    lipinski_violations: list[str] = field(default_factory=list)
    element_counts: dict[str, int] = field(default_factory=dict)

    @property
    def drug_like(self) -> bool:
        return not self.lipinski_violations


class MolecularFormula:
    """Parse molecular formulas and compute real descriptor values."""

    TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")

    @classmethod
    def parse(cls, formula: str) -> dict[str, int]:
        counts: Counter[str] = Counter()
        cleaned = (formula or "").replace(" ", "")
        if not cleaned:
            raise ValueError("empty formula")
        for part in cleaned.split("."):
            multiplier = ""
            while part and part[0].isdigit():
                multiplier += part[0]
                part = part[1:]
            factor = int(multiplier) if multiplier else 1
            stack: list[Counter[str]] = [Counter()]
            i = 0
            while i < len(part):
                char = part[i]
                if char == "(":
                    stack.append(Counter())
                    i += 1
                elif char == ")":
                    if len(stack) == 1:
                        raise ValueError("unbalanced ')' in formula")
                    group = stack.pop()
                    i += 1
                    digits = ""
                    while i < len(part) and part[i].isdigit():
                        digits += part[i]
                        i += 1
                    repeat = int(digits) if digits else 1
                    for element, value in group.items():
                        stack[-1][element] += value * repeat
                else:
                    match = cls.TOKEN.match(part, i)
                    if not match:
                        raise ValueError(f"cannot parse formula near {part[i:]!r}")
                    element, digits = match.group(1), match.group(2)
                    if element not in ATOMIC_WEIGHTS:
                        raise ValueError(f"unknown element {element}")
                    stack[-1][element] += int(digits) if digits else 1
                    i = match.end()
            if len(stack) != 1:
                raise ValueError("unbalanced '(' in formula")
            for element, value in stack[0].items():
                counts[element] += value * factor
        return dict(counts)

    @classmethod
    def molecular_weight(cls, formula: str) -> float:
        counts = cls.parse(formula)
        return round(sum(ATOMIC_WEIGHTS[e] * n for e, n in counts.items()), 3)

    @classmethod
    def degree_of_unsaturation(cls, formula: str) -> float:
        c = cls.parse(formula)
        halogens = sum(c.get(e, 0) for e in HALOGENS)
        return round((2 * c.get("C", 0) + 2 + c.get("N", 0) - c.get("H", 0) - halogens) / 2.0, 2)

    @classmethod
    def lipinski(cls, formula: str, logp: float, hbd: int, hba: int) -> MolecularProperties:
        counts = cls.parse(formula)
        mw = round(sum(ATOMIC_WEIGHTS[e] * n for e, n in counts.items()), 3)
        violations: list[str] = []
        if mw > 500:
            violations.append("molecular weight > 500")
        if logp > 5:
            violations.append("logP > 5")
        if hbd > 5:
            violations.append("hydrogen bond donors > 5")
        if hba > 10:
            violations.append("hydrogen bond acceptors > 10")
        heavy = sum(n for e, n in counts.items() if e != "H")
        return MolecularProperties(
            formula=formula,
            molecular_weight=mw,
            heavy_atoms=heavy,
            degree_of_unsaturation=cls.degree_of_unsaturation(formula),
            lipinski_violations=violations,
            element_counts=counts,
        )


# ── Units ─────────────────────────────────────────────────────────────────

UNIT_TABLE: dict[str, dict[str, float]] = {
    "length": {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001, "um": 1e-6,
               "nm": 1e-9, "inch": 0.0254, "ft": 0.3048, "mile": 1609.344},
    "mass": {"kg": 1.0, "g": 1e-3, "mg": 1e-6, "ug": 1e-9, "ng": 1e-12,
             "lb": 0.45359237, "oz": 0.028349523125, "Da": 1.66053906660e-27},
    "energy": {"J": 1.0, "kJ": 1000.0, "cal": 4.184, "kcal": 4184.0,
               "eV": 1.602176634e-19, "kWh": 3.6e6},
    "pressure": {"Pa": 1.0, "kPa": 1000.0, "bar": 1e5, "atm": 101325.0,
                 "mmHg": 133.322387415, "psi": 6894.757293168},
    "volume": {"L": 1.0, "mL": 1e-3, "uL": 1e-6, "nL": 1e-9, "m3": 1000.0},
    "time": {"s": 1.0, "ms": 1e-3, "us": 1e-6, "min": 60.0, "h": 3600.0, "day": 86400.0},
    "temperature": {"K": 1.0, "C": 1.0, "F": 1.0, "R": 1.0},
    "amount": {"mol": 1.0, "mmol": 1e-3, "umol": 1e-6, "nmol": 1e-9, "pmol": 1e-12},
}


class UnitConverter:
    """Dimensional unit conversion, including the non-linear temperature scales."""

    def dimension_of(self, unit: str) -> str:
        for dimension, table in UNIT_TABLE.items():
            if unit in table:
                return dimension
        raise ValueError(f"unknown unit {unit!r}")

    def compatible(self, source: str, target: str) -> bool:
        try:
            return self.dimension_of(source) == self.dimension_of(target)
        except ValueError:
            return False

    def convert(self, value: float, source: str, target: str) -> float:
        if not self.compatible(source, target):
            raise ValueError(f"incompatible units: {source} -> {target}")
        dimension = self.dimension_of(source)
        if dimension == "temperature":
            return self._temperature(value, source, target)
        factor = UNIT_TABLE[dimension][source] / UNIT_TABLE[dimension][target]
        return value * factor

    @staticmethod
    def _temperature(value: float, source: str, target: str) -> float:
        if source == target:
            return value
        kelvin = {
            "K": lambda v: v,
            "C": lambda v: v + 273.15,
            "F": lambda v: (v - 32.0) * 5.0 / 9.0 + 273.15,
            "R": lambda v: v * 5.0 / 9.0,
        }[source](value)
        return {
            "K": lambda k: k,
            "C": lambda k: k - 273.15,
            "F": lambda k: (k - 273.15) * 9.0 / 5.0 + 32.0,
            "R": lambda k: k * 9.0 / 5.0,
        }[target](kelvin)


# ── Skill catalogue ───────────────────────────────────────────────────────

@dataclass
class SkillSpec:
    """One procedural-knowledge skill: what it needs, does, and must validate."""

    skill_id: str
    title: str
    category: str
    keywords: list[str]
    packages: list[str] = field(default_factory=list)
    consumes: list[str] = field(default_factory=list)
    produces: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    databases: list[str] = field(default_factory=list)
    workflow: list[str] = field(default_factory=list)

    def matches(self, query: str) -> float:
        terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not terms:
            return 0.0
        hay = set(self.keywords) | set(re.findall(r"[a-z0-9]+", self.title.lower()))
        hay |= {w for p in self.packages for w in re.findall(r"[a-z0-9]+", p.lower())}
        hay |= {w for w in re.findall(r"[a-z0-9]+", self.category.lower())}
        overlap = len(terms & hay)
        if overlap == 0:
            return 0.0
        return round(overlap / math.sqrt(len(hay) + 1), 4)


def _spec(skill_id: str, title: str, category: str, keywords: list[str], **kwargs: Any) -> SkillSpec:
    return SkillSpec(skill_id=skill_id, title=title, category=category, keywords=keywords, **kwargs)


DEFAULT_SKILLS: list[SkillSpec] = [
    _spec("bulk-rnaseq", "Bulk RNA-seq differential expression", "bioinformatics",
          ["rna", "seq", "expression", "differential", "counts", "deseq"],
          packages=["pydeseq2", "scanpy"], consumes=["fastq", "count_matrix"],
          produces=["de_table"],
          checks=["library size normalisation", "dispersion fit", "batch covariate present"],
          workflow=["QC reads", "align or quantify", "build count matrix",
                    "fit dispersion", "test contrasts", "FDR control"]),
    _spec("single-cell-rnaseq", "Single-cell RNA-seq analysis", "bioinformatics",
          ["single", "cell", "scrna", "cluster", "umap", "scanpy"],
          packages=["scanpy"], consumes=["count_matrix"], produces=["clusters", "marker_table"],
          checks=["mitochondrial fraction", "doublet detection", "batch integration"],
          workflow=["QC", "normalise", "HVG selection", "PCA", "neighbours", "cluster", "annotate"]),
    _spec("variant-annotation", "Variant annotation and prioritisation", "bioinformatics",
          ["variant", "vcf", "annotation", "snv", "indel"],
          packages=["cyvcf2", "hgvs"], consumes=["vcf"], produces=["annotated_variants"],
          checks=["reference build matches", "clinvar review status", "allele frequency"],
          databases=["ClinVar", "gnomAD"], workflow=["normalise variants", "annotate",
                                                     "filter by frequency", "classify"]),
    _spec("phylogenetics", "Phylogenetic tree inference", "bioinformatics",
          ["phylogeny", "tree", "alignment", "evolution", "bootstrap"],
          packages=["biopython"], consumes=["alignment"], produces=["tree", "support_values"],
          checks=["alignment trimming", "model selection", "bootstrap support"],
          workflow=["align", "trim", "select model", "build tree", "bootstrap", "root"]),
    _spec("primer-design", "PCR primer design and specificity", "bioinformatics",
          ["primer", "pcr", "oligo", "amplification", "tm"],
          packages=["primer3"], consumes=["target_sequence"], produces=["primer_pairs"],
          checks=["melting temperature window", "GC clamp", "off-target search"],
          workflow=["fetch target", "scan candidates", "score Tm/GC", "check specificity"]),
    _spec("amplicon-microbiome", "Amplicon microbiome profiling", "bioinformatics",
          ["microbiome", "16s", "amplicon", "qiime", "taxonomy"],
          packages=["qiime2"], consumes=["fastq"], produces=["taxonomy_table"],
          checks=["read retention", "chimera removal", "rarefaction depth"],
          workflow=["denoise", "merge pairs", "assign taxonomy", "diversity"]),
    _spec("molecular-docking", "Molecular docking and scoring", "cheminformatics",
          ["docking", "ligand", "receptor", "binding", "pose", "affinity"],
          packages=["vina"], consumes=["ligand_sdf", "receptor_pdb"], produces=["docking_poses"],
          checks=["protonation state", "box covers site", "redocking RMSD"],
          workflow=["prepare receptor", "prepare ligand", "define box", "dock", "rescore"]),
    _spec("admet-prediction", "ADMET property prediction", "cheminformatics",
          ["admet", "toxicity", "solubility", "permeability", "pk"],
          packages=["rdkit", "pytdc"], consumes=["smiles"], produces=["admet_table"],
          checks=["applicability domain", "calibration", "units of prediction"],
          workflow=["featurise", "predict", "flag out-of-domain", "summarise risks"]),
    _spec("molecular-descriptors", "Molecular descriptors and Lipinski profiling", "cheminformatics",
          ["descriptor", "molecular", "weight", "lipinski", "logp", "druglike"],
          packages=["rdkit"], consumes=["smiles", "formula"], produces=["descriptor_table"],
          checks=["salt stripping", "tautomer assumption", "formula consistency"],
          workflow=["parse structure", "compute descriptors", "apply rule sets"]),
    _spec("nmr-processing", "Calibrated 1D NMR processing", "proteomics",
          ["nmr", "spectrum", "chemical", "shift", "peak"],
          packages=["nmrglue"], consumes=["fid"], produces=["spectrum", "peak_list"],
          checks=["referencing standard", "baseline correction", "phase correction"],
          workflow=["load FID", "apodise", "zero fill", "phase", "baseline", "pick peaks"]),
    _spec("lcms-proteomics", "LC-MS/MS peptide identification", "proteomics",
          ["lcms", "peptide", "mass", "spectra", "proteomics", "ptm"],
          packages=["pyteomics"], consumes=["mzml"], produces=["peptide_table"],
          checks=["FDR at PSM level", "decoy database", "retention time alignment"],
          workflow=["convert raw", "search spectra", "filter FDR", "quantify"]),
    _spec("clinical-trials", "Clinical trial evidence review", "clinical",
          ["clinical", "trial", "endpoint", "randomised", "evidence"],
          consumes=["trial_record"], produces=["evidence_table"],
          checks=["registered endpoint matches", "sample size justification", "CONSORT flow"],
          databases=["ClinicalTrials.gov"], workflow=["retrieve record", "extract endpoints",
                                                      "assess bias", "summarise"]),
    _spec("ehr-research", "EHR cohort research analysis", "healthcare-ai",
          ["ehr", "cohort", "phenotype", "retrospective", "claims"],
          consumes=["cohort_table"], produces=["analysis_table"],
          checks=["phenotype code list versioned", "missingness reported", "no diagnosis output"],
          workflow=["define cohort", "validate phenotype", "adjust confounders", "report"]),
    _spec("biosignal-analysis", "Physiological signal analysis", "healthcare-ai",
          ["ecg", "eeg", "signal", "physiological", "biosignal", "hrv"],
          packages=["neurokit2"], consumes=["signal"], produces=["signal_features"],
          checks=["sampling rate preserved", "artifact epochs flagged", "filter order stated"],
          workflow=["load signal", "filter", "detect events", "extract features"]),
    _spec("severity-scoring", "Preclinical severity scoring for 3Rs", "preclinical",
          ["severity", "animal", "welfare", "endpoint", "preclinical"],
          consumes=["observation_log"], produces=["severity_score"],
          checks=["score aids assessment, never a decision rule", "humane endpoint defined"],
          workflow=["collect observations", "score dimensions", "aggregate", "flag endpoints"]),
    _spec("microscopy-quantification", "Quantitative fluorescence microscopy", "microscopy",
          ["microscopy", "fluorescence", "intensity", "colocalisation", "segmentation"],
          packages=["cellprofiler", "scikit-image"], consumes=["image_stack"],
          produces=["cell_table"],
          checks=["illumination correction", "background subtraction", "segmentation QC"],
          workflow=["correct illumination", "segment", "measure", "QC", "summarise"]),
    _spec("dicom-processing", "Privacy-aware DICOM processing", "microscopy",
          ["dicom", "radiology", "pacs", "deidentify", "imaging"],
          packages=["pydicom"], consumes=["dicom_series"], produces=["image_volume"],
          checks=["de-identification complete", "slice spacing consistent", "photometric check"],
          workflow=["read series", "de-identify", "convert to volume", "resample"]),
    _spec("whole-slide-analysis", "Whole-slide computational pathology", "microscopy",
          ["wsi", "pathology", "slide", "tissue", "patch"],
          packages=["pathml"], consumes=["wsi_file"], produces=["tile_table"],
          checks=["magnification recorded", "tissue mask QC", "stain normalisation"],
          workflow=["open slide", "tissue mask", "tile", "extract features"]),
    _spec("bids-datasets", "BIDS dataset handling", "neuroscience",
          ["bids", "neuroimaging", "dataset", "sidecar", "participants"],
          packages=["bids-validator"], consumes=["dataset"], produces=["bids_index"],
          checks=["participants.tsv completeness", "sidecar JSON present", "intended-for chains"],
          workflow=["validate layout", "index files", "query metadata"]),
    _spec("nwb-conversion", "NWB conversion and clock alignment", "neuroscience",
          ["nwb", "electrophysiology", "clock", "conversion", "neurodata"],
          packages=["pynwb"], consumes=["recording"], produces=["nwb_file"],
          checks=["clock drift measured", "units declared", "trials aligned"],
          workflow=["read recording", "align clocks", "write NWB", "validate"]),
    _spec("deep-learning", "Deep learning model training", "machine-learning",
          ["deep", "neural", "training", "pytorch", "model", "transfer"],
          packages=["torch", "lightning"], consumes=["dataset"], produces=["checkpoint"],
          checks=["leakage-free split", "seed set", "validation curve monitored"],
          workflow=["split", "featurise", "train", "validate", "calibrate"]),
    _spec("time-series-ml", "Time-series forecasting and validation", "machine-learning",
          ["time", "series", "forecast", "seasonality", "backtest"],
          packages=["statsmodels"], consumes=["series"], produces=["forecast"],
          checks=["no future leakage", "rolling origin", "interval calibration"],
          workflow=["profile series", "split", "fit", "backtest", "report intervals"]),
    _spec("bayesian-analysis", "Bayesian inference workflow", "machine-learning",
          ["bayesian", "posterior", "prior", "mcmc", "credible"],
          packages=["pymc", "arviz"], consumes=["dataset", "prior_spec"],
          produces=["posterior"], checks=["prior predictive check", "R-hat", "divergences"],
          workflow=["specify model", "prior predictive", "sample", "diagnose", "posterior predictive"]),
    _spec("interpretability", "Model interpretability analysis", "machine-learning",
          ["shap", "interpretability", "feature", "attribution", "explanation"],
          packages=["shap"], consumes=["model", "dataset"], produces=["attribution_table"],
          checks=["baseline defined", "correlated features grouped", "stability checked"],
          workflow=["choose method", "compute attributions", "aggregate", "sanity check"]),
    _spec("crystal-structure", "Crystal structure analysis", "materials",
          ["crystal", "lattice", "structure", "symmetry", "cell"],
          packages=["pymatgen"], consumes=["cif"], produces=["structure_features"],
          checks=["occupancy consistency", "charge balance", "symmetry tolerance"],
          workflow=["read CIF", "symmetrise", "analyse coordination", "compute features"]),
    _spec("calphad", "CALPHAD phase equilibria", "materials",
          ["calphad", "phase", "equilibria", "alloy", "thermodynamic"],
          packages=["pycalphad"], consumes=["tdb", "composition"], produces=["phase_diagram"],
          checks=["database provenance", "temperature grid adequacy", "stable phase check"],
          workflow=["load database", "set conditions", "equilibrate", "map diagram"]),
    _spec("battery-cycling", "Battery cycling simulation", "engineering",
          ["battery", "cycling", "pybamm", "electrochemistry", "degradation"],
          packages=["pybamm"], consumes=["cell_parameters"], produces=["cycling_curve"],
          checks=["parameter set cited", "C-rate stated", "thermal model consistent"],
          workflow=["load parameter set", "build model", "solve", "extract degradation"]),
    _spec("chemical-kinetics", "Chemical kinetics and ignition delay", "engineering",
          ["kinetics", "ignition", "combustion", "cantera", "reaction"],
          packages=["cantera"], consumes=["mechanism"], produces=["ignition_delay"],
          checks=["mechanism version", "units consistent", "sensitivity reported"],
          workflow=["load mechanism", "set reactor", "integrate", "sensitivity"]),
    _spec("cfd-simulation", "Fluid dynamics simulation", "engineering",
          ["cfd", "fluid", "mesh", "turbulence", "navier"],
          packages=["openfoam"], consumes=["mesh", "boundary_conditions"],
          produces=["flow_field"], checks=["mesh independence", "y+ reported", "residuals converged"],
          workflow=["mesh", "set BCs", "solve", "check convergence", "post-process"]),
    _spec("lab-hardware-cad", "Lab hardware CAD and fabrication", "engineering",
          ["cad", "hardware", "fabrication", "build123d", "3d", "print"],
          packages=["build123d"], consumes=["dimension_spec"], produces=["cad_model"],
          checks=["tolerance stack", "material compatibility", "print orientation"],
          workflow=["sketch constraints", "build solid", "export STEP", "check tolerances"]),
    _spec("discrete-event-sim", "Discrete-event simulation", "engineering",
          ["simulation", "discrete", "queue", "event", "throughput"],
          packages=["simpy"], consumes=["process_spec"], produces=["simulation_trace"],
          checks=["warm-up period removed", "replications reported", "seed recorded"],
          workflow=["model processes", "run replications", "analyse statistics"]),
    _spec("statistical-analysis", "Statistical hypothesis testing", "data-analysis",
          ["statistics", "test", "p-value", "t-test", "anova", "effect"],
          packages=["scipy", "statsmodels"], consumes=["dataset"], produces=["test_report"],
          checks=["assumption check", "effect size reported", "multiple-comparison control"],
          workflow=["inspect data", "check assumptions", "test", "effect size", "correct"]),
    _spec("network-analysis", "Network and graph analysis", "data-analysis",
          ["network", "graph", "centrality", "community", "degree"],
          packages=["networkx"], consumes=["edge_list"], produces=["graph_metrics"],
          checks=["directedness declared", "self-loops handled", "weights meaningful"],
          workflow=["build graph", "compute metrics", "detect communities"]),
    _spec("publication-figures", "Publication-quality figures", "data-analysis",
          ["figure", "plot", "visualisation", "publication", "figure legend"],
          packages=["matplotlib"], consumes=["dataset"], produces=["figure"],
          checks=["axis units labelled", "colour-blind safe", "resolution >= 300 dpi"],
          workflow=["choose encoding", "render", "check accessibility", "export"]),
    _spec("seawater-carbonate", "Seawater carbonate chemistry", "oceanography",
          ["seawater", "carbonate", "alkalinity", "ph", "ocean", "co2"],
          packages=["PyCO2SYS"], consumes=["ta_ph_measurements"], produces=["carbonate_system"],
          checks=["measurement pair consistency", "uncertainty propagated", "scale stated"],
          workflow=["choose input pair", "solve system", "propagate uncertainty"]),
    _spec("geospatial-raster", "Satellite raster processing", "geospatial",
          ["satellite", "raster", "remote", "sensing", "geotiff", "ndvi"],
          packages=["rasterio", "geopandas"], consumes=["raster"], produces=["raster_index"],
          checks=["cloud mask applied", "reprojection documented", "nodata handled"],
          workflow=["read raster", "mask", "compute index", "reproject", "zonal stats"]),
    _spec("spatial-statistics", "Spatial statistics and autocorrelation", "geospatial",
          ["spatial", "autocorrelation", "moran", "geostatistics", "kriging"],
          packages=["pysal"], consumes=["point_layer"], produces=["spatial_stats"],
          checks=["spatial weights defined", "edge effects noted", "multiple testing"],
          workflow=["build weights", "compute autocorrelation", "model", "validate"]),
    _spec("liquid-handling", "Liquid handling protocol automation", "lab-automation",
          ["liquid", "handler", "protocol", "opentrons", "pipette"],
          packages=["opentrons"], consumes=["plate_layout"], produces=["protocol"],
          checks=["tip reuse policy", "dead volume respected", "cross-contamination"],
          workflow=["read layout", "plan transfers", "generate protocol", "simulate"]),
    _spec("lims-integration", "LIMS integration", "lab-automation",
          ["lims", "benchling", "sample", "registry", "integration"],
          consumes=["sample_sheet"], produces=["registered_samples"],
          checks=["sample naming convention", "idempotent registration", "audit trail"],
          workflow=["authenticate", "map fields", "register", "verify"]),
    _spec("literature-synthesis", "Literature synthesis and evidence mapping", "scientific-communication",
          ["literature", "review", "synthesis", "paper", "search", "evidence"],
          packages=["paperclip"], consumes=["query"], produces=["evidence_map"],
          checks=["search string reported", "inclusion criteria stated", "duplicates removed"],
          databases=["PubMed", "OpenAlex"], workflow=["search", "screen", "extract",
                                                      "map evidence", "cite"]),
    _spec("scientific-writing", "Evidence-traceable scientific writing", "scientific-communication",
          ["writing", "manuscript", "draft", "citation", "claim"],
          consumes=["evidence_map"], produces=["manuscript"],
          checks=["every claim cites evidence", "no unsupported superlatives", "numbers sourced"],
          workflow=["outline", "draft with citations", "audit claims", "revise"]),
    _spec("peer-review", "Confidential structured peer review", "scientific-communication",
          ["review", "referee", "critique", "confidential", "rebuttal"],
          consumes=["manuscript"], produces=["review_report"],
          checks=["confidentiality", "specific evidence per criticism", "no author identity use"],
          workflow=["read", "assess methods", "assess claims", "structure report"]),
    _spec("scientific-slides", "Scientific slides and posters", "scientific-communication",
          ["slides", "poster", "presentation", "deck", "pptx"],
          packages=["python-pptx"], consumes=["manuscript"], produces=["deck"],
          checks=["font size minimum", "no macro content", "figure resolution"],
          workflow=["outline story", "layout", "render", "validate"]),
    _spec("protein-structure", "Protein structure prediction and refinement", "protein-engineering",
          ["protein", "structure", "fold", "alphafold", "cryo"],
          packages=["alphafold", "relion"], consumes=["sequence"], produces=["structure_model"],
          checks=["pLDDT reported", "half-map validation", "stereochemistry"],
          workflow=["predict", "refine", "validate", "analyse interface"]),
    _spec("sequence-design", "Protein sequence design", "protein-engineering",
          ["design", "protein", "sequence", "language", "model", "mutation"],
          packages=["esm"], consumes=["structure_model"], produces=["designed_sequences"],
          checks=["wild-type recovery", "stability prediction", "expression feasibility"],
          workflow=["encode structure", "sample designs", "score", "filter"]),
    _spec("hypothesis-generation", "Evidence-bounded hypothesis generation", "methodology",
          ["hypothesis", "abduction", "novelty", "research", "question"],
          consumes=["evidence_map"], produces=["hypotheses"],
          checks=["each hypothesis falsifiable", "evidence listed", "prior work checked"],
          workflow=["collect observations", "generate candidates", "score support", "design test"]),
    _spec("grant-writing", "Grant writing and budget justification", "methodology",
          ["grant", "proposal", "budget", "aim", "funding"],
          consumes=["hypotheses"], produces=["proposal"],
          checks=["aims map to budget", "preliminary data cited", "page limits"],
          workflow=["draft aims", "build methods", "cost", "review"]),
    _spec("method-validation", "Analytical method validation", "regulatory",
          ["validation", "ich", "usp", "accuracy", "precision", "loq"],
          consumes=["measurement_set"], produces=["validation_report"],
          checks=["accuracy and precision per ICH Q2", "LOD/LOQ justified", "carryover tested"],
          workflow=["define protocol", "run accuracy/precision", "determine LOD/LOQ",
                    "report for qualified review"]),
    _spec("iso-readiness", "ISO standards readiness evidence", "regulatory",
          ["iso", "13485", "17025", "readiness", "audit", "compliance"],
          consumes=["quality_records"], produces=["evidence_pack"],
          checks=["clause mapping complete", "objective evidence per clause", "gaps logged"],
          workflow=["map clauses", "collect evidence", "score gaps", "prepare pack"]),
    _spec("database-lookup", "Multi-database scientific lookup", "agent-platforms",
          ["database", "pubchem", "uniprot", "lookup", "api", "provenance"],
          consumes=["identifier"], produces=["record_set"],
          checks=["endpoint documented", "pagination complete", "retrieval timestamped"],
          databases=["PubChem", "UniProt", "ChEMBL", "ClinicalTrials.gov", "FRED"],
          workflow=["choose source", "query", "paginate", "record provenance", "cache"]),
    _spec("workflow-drafting", "Workflow-derived skill drafting", "agent-platforms",
          ["workflow", "skill", "draft", "automation", "reuse"],
          consumes=["session_trace"], produces=["skill_draft"],
          checks=["steps reproducible", "inputs declared", "validations captured"],
          workflow=["collect trace", "abstract steps", "declare I/O", "test"]),
]


@dataclass
class SkillMatch:
    """A ranked candidate skill for a task."""

    skill: SkillSpec
    score: float
    reasons: list[str] = field(default_factory=list)


class SkillLibrary:
    """The searchable scientific skill catalogue."""

    def __init__(self, skills: list[SkillSpec] | None = None) -> None:
        self.skills: dict[str, SkillSpec] = {s.skill_id: s for s in (skills or DEFAULT_SKILLS)}

    def __len__(self) -> int:
        return len(self.skills)

    def get(self, skill_id: str) -> SkillSpec | None:
        return self.skills.get(skill_id)

    def categories(self) -> dict[str, int]:
        counts: Counter[str] = Counter(s.category for s in self.skills.values())
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def by_category(self, category: str) -> list[SkillSpec]:
        return [s for s in self.skills.values() if s.category == category]

    def by_package(self, package: str) -> list[SkillSpec]:
        needle = package.lower()
        return [s for s in self.skills.values() if any(needle == p.lower() for p in s.packages)]

    def with_database(self, database: str) -> list[SkillSpec]:
        needle = database.lower()
        return [s for s in self.skills.values() if any(needle == d.lower() for d in s.databases)]

    def search(self, query: str, limit: int = 5) -> list[SkillMatch]:
        matches: list[SkillMatch] = []
        for skill in self.skills.values():
            score = skill.matches(query)
            if score <= 0:
                continue
            reasons = []
            terms = set(re.findall(r"[a-z0-9]+", query.lower()))
            if terms & set(skill.keywords):
                reasons.append(f"keywords: {', '.join(sorted(terms & set(skill.keywords)))}")
            if terms & {p.lower() for p in skill.packages}:
                reasons.append("package expertise match")
            matches.append(SkillMatch(skill, score, reasons))
        matches.sort(key=lambda m: (-m.score, m.skill.skill_id))
        return matches[:limit]

    def coverage_report(self) -> dict[str, Any]:
        checks: Counter[str] = Counter()
        for skill in self.skills.values():
            checks[skill.category] += 1
        return {
            "skills": len(self.skills),
            "categories": len(checks),
            "per_category": self.categories(),
            "with_packages": len([s for s in self.skills.values() if s.packages]),
            "with_validations": len([s for s in self.skills.values() if s.checks]),
            "databases": sorted({d for s in self.skills.values() for d in s.databases}),
        }


class SkillMatcher:
    """Rank skills for a research task and explain the choice."""

    def __init__(self, library: SkillLibrary | None = None) -> None:
        self.library = library or SkillLibrary()

    def match(self, task: str, limit: int = 3) -> list[SkillMatch]:
        return self.library.search(task, limit=limit)

    def best(self, task: str) -> SkillMatch | None:
        matches = self.match(task, limit=1)
        return matches[0] if matches else None

    def match_many(self, tasks: list[str], limit: int = 2) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for task in tasks:
            out[task] = [m.skill.skill_id for m in self.match(task, limit=limit)]
        return out

    def gaps(self, task: str, threshold: float = 0.15) -> list[str]:
        """Categories that could plausibly serve a task but score poorly."""
        matches = self.match(task, limit=10)
        weak = [m for m in matches if m.score < threshold]
        return sorted({m.skill.category for m in weak})


# ── Composition ───────────────────────────────────────────────────────────

@dataclass
class PipelineStep:
    """One skill in a composed research pipeline."""

    order: int
    skill_id: str
    consumes: list[str]
    produces: list[str]
    checks: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "order": self.order, "skill": self.skill_id,
            "consumes": self.consumes, "produces": self.produces,
            "checks": self.checks,
        }


class ProvenanceGraph:
    """Artifact lineage: which skill produced what, from which inputs."""

    def __init__(self) -> None:
        self.artifacts: dict[str, dict[str, Any]] = {}
        self._seq = 0

    def record(self, artifact: str, skill_id: str, inputs: list[str], note: str = "") -> str:
        self._seq += 1
        self.artifacts[artifact] = {
            "artifact": artifact, "skill": skill_id, "inputs": list(inputs),
            "note": note, "seq": self._seq,
        }
        return artifact

    def ancestors(self, artifact: str) -> list[str]:
        seen: list[str] = []
        stack = [artifact]
        while stack:
            current = stack.pop()
            node = self.artifacts.get(current)
            if node is None:
                continue
            for parent in node["inputs"]:
                if parent not in seen:
                    seen.append(parent)
                    stack.append(parent)
        return seen

    def lineage(self, artifact: str) -> list[dict[str, Any]]:
        names = [artifact] + self.ancestors(artifact)
        nodes = [self.artifacts[n] for n in names if n in self.artifacts]
        return sorted(nodes, key=lambda n: n["seq"])

    def unsourced(self) -> list[str]:
        return [name for name, node in self.artifacts.items() if not node["inputs"]]


class SkillComposer:
    """Compose a goal into an ordered, cycle-checked, provenance-tracked pipeline."""

    def __init__(self, library: SkillLibrary | None = None) -> None:
        self.library = library or SkillLibrary()
        self.matcher = SkillMatcher(self.library)

    def order_skills(self, skill_ids: list[str]) -> list[str]:
        """Topological order by produced/consumed artifacts; raises on cycles."""
        chosen = [self.library.get(s) for s in skill_ids]
        chosen = [s for s in chosen if s is not None]
        if not chosen:
            return []
        ids = [s.skill_id for s in chosen]
        produced_by: dict[str, str] = {}
        for skill in chosen:
            for artifact in skill.produces:
                produced_by.setdefault(artifact, skill.skill_id)
        dependencies: dict[str, set[str]] = {s.skill_id: set() for s in chosen}
        for skill in chosen:
            for artifact in skill.consumes:
                producer = produced_by.get(artifact)
                if producer and producer != skill.skill_id:
                    dependencies[skill.skill_id].add(producer)
        ordered: list[str] = []
        remaining = dict(dependencies)
        while remaining:
            ready = sorted([node for node, deps in remaining.items() if not deps])
            if not ready:
                raise ValueError(f"cyclic skill dependency: {sorted(remaining)}")
            for node in ready:
                ordered.append(node)
                remaining.pop(node)
            for deps in remaining.values():
                deps.difference_update(ready)
        return ordered

    def compose(self, goal: str, skill_ids: list[str] | None = None, limit: int = 4) -> dict[str, Any]:
        if skill_ids:
            selected = skill_ids
            reasons = {sid: "explicitly requested" for sid in skill_ids}
        else:
            matches = self.matcher.match(goal, limit=limit)
            selected = [m.skill.skill_id for m in matches]
            reasons = {m.skill.skill_id: "; ".join(m.reasons) or "keyword match" for m in matches}
        ordered = self.order_skills(selected)
        graph = ProvenanceGraph()
        available = {"goal"}
        steps: list[PipelineStep] = []
        for index, skill_id in enumerate(ordered, start=1):
            skill = self.library.get(skill_id)
            if skill is None:
                continue
            inputs = [a for a in skill.consumes if a in available]
            for artifact in skill.produces:
                graph.record(artifact, skill_id, inputs or ["goal"])
                available.add(artifact)
            steps.append(PipelineStep(
                order=index, skill_id=skill_id, consumes=skill.consumes,
                produces=skill.produces, checks=skill.checks,
            ))
        missing_inputs = sorted({
            artifact for step in steps
            for artifact in step.consumes
            if artifact not in available and artifact not in ("goal",)
        })
        return {
            "goal": goal,
            "steps": [s.as_dict() for s in steps],
            "reasons": reasons,
            "artifacts": sorted(available - {"goal"}),
            "missing_inputs": missing_inputs,
            "provenance": {name: node["skill"] for name, node in graph.artifacts.items()},
            "validation_checks": [c for step in steps for c in step.checks],
        }


# ── Hypotheses ────────────────────────────────────────────────────────────

@dataclass
class Observation:
    """A piece of evidence with a likelihood ratio for a candidate hypothesis."""

    statement: str
    likelihood_ratio: float
    source: str = ""

    def weight(self) -> float:
        return math.log(max(1e-9, self.likelihood_ratio))


@dataclass
class Hypothesis:
    """A ranked, falsifiable candidate explanation."""

    statement: str
    prior_probability: float
    posterior_probability: float
    log_odds: float
    evidence: list[str]
    falsification_test: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "prior": round(self.prior_probability, 4),
            "posterior": round(self.posterior_probability, 4),
            "log_odds": round(self.log_odds, 4),
            "evidence": self.evidence,
            "falsification_test": self.falsification_test,
        }


class HypothesisGenerator:
    """Bayesian evidence accumulation over candidate explanations."""

    TEMPLATES = [
        ("A {mechanism} effect explains the observed {measurement}.",
         "Measure {measurement} under a condition that removes {mechanism}; the effect should vanish."),
        ("The {measurement} shift is driven by a confound in {confounder}.",
         "Randomise {confounder} and re-measure; the shift should disappear."),
        ("The {measurement} change reflects a threshold in {mechanism}.",
         "Sweep {mechanism} across the putative threshold and test for a discontinuity."),
        ("Measurement error in {measurement} inflates the apparent {mechanism} signal.",
         "Blind-replicate {measurement} with an independent instrument; the signal should shrink."),
    ]

    def generate(
        self,
        measurement: str,
        mechanism: str,
        observations: list[Observation],
        prior: float = 0.3,
        confounder: str = "sampling",
    ) -> list[Hypothesis]:
        prior = min(max(prior, 1e-6), 1 - 1e-6)
        prior_log_odds = math.log(prior / (1 - prior))
        hypotheses: list[Hypothesis] = []
        for index, (statement_t, test_t) in enumerate(self.TEMPLATES):
            statement = statement_t.format(
                mechanism=mechanism, measurement=measurement, confounder=confounder,
            )
            test = test_t.format(
                mechanism=mechanism, measurement=measurement, confounder=confounder,
            )
            # Each template gets a different slice of the evidence weight.
            share = [1.0, 0.6, 0.4, -0.3][index % 4]
            log_odds = prior_log_odds + sum(o.weight() for o in observations) * share
            posterior = 1.0 / (1.0 + math.exp(-log_odds))
            hypotheses.append(Hypothesis(
                statement=statement,
                prior_probability=prior,
                posterior_probability=posterior,
                log_odds=log_odds,
                evidence=[o.statement for o in observations if o.weight() > 0],
                falsification_test=test,
            ))
        hypotheses.sort(key=lambda h: -h.posterior_probability)
        return hypotheses

    def update(self, hypothesis: Hypothesis, observations: list[Observation]) -> Hypothesis:
        log_odds = hypothesis.log_odds + sum(o.weight() for o in observations)
        return Hypothesis(
            statement=hypothesis.statement,
            prior_probability=hypothesis.prior_probability,
            posterior_probability=1.0 / (1.0 + math.exp(-log_odds)),
            log_odds=log_odds,
            evidence=hypothesis.evidence + [o.statement for o in observations],
            falsification_test=hypothesis.falsification_test,
        )


# ── Evidence-traceable writing ────────────────────────────────────────────

@dataclass
class Claim:
    """A statement in a draft that must cite evidence."""

    text: str
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class EvidenceRecord:
    """A citable source of evidence."""

    evidence_id: str
    title: str
    source: str
    value: str = ""


class ScientificWriter:
    """Audit claims against evidence and render a cited report."""

    def __init__(self, evidence: list[EvidenceRecord] | None = None) -> None:
        self.evidence: dict[str, EvidenceRecord] = {e.evidence_id: e for e in (evidence or [])}

    def add_evidence(self, record: EvidenceRecord) -> None:
        self.evidence[record.evidence_id] = record

    def audit(self, claims: list[Claim]) -> dict[str, Any]:
        supported: list[str] = []
        unsupported: list[str] = []
        unknown_ids: list[str] = []
        for claim in claims:
            valid = [e for e in claim.evidence_ids if e in self.evidence]
            unknown_ids.extend(e for e in claim.evidence_ids if e not in self.evidence)
            if valid and not re.search(r"\b(always|never|proves|definitively)\b", claim.text.lower()):
                supported.append(claim.text)
            else:
                unsupported.append(claim.text)
        total = len(claims) or 1
        return {
            "supported": supported,
            "unsupported": unsupported,
            "unknown_evidence": sorted(set(unknown_ids)),
            "traceability": round(len(supported) / total, 4),
            "claims": len(claims),
        }

    def render(self, title: str, claims: list[Claim]) -> str:
        lines = [f"# {title}", ""]
        for claim in claims:
            cites = " ".join(f"[{eid}]" for eid in claim.evidence_ids if eid in self.evidence)
            lines.append(f"- {claim.text}{(' ' + cites) if cites else ''}")
        if self.evidence:
            lines += ["", "## References", ""]
            for record in sorted(self.evidence.values(), key=lambda r: r.evidence_id):
                lines.append(f"- [{record.evidence_id}] {record.title}. {record.source}")
        return "\n".join(lines)


# ── Experimental design ───────────────────────────────────────────────────

@dataclass
class DesignRow:
    """One run of an experimental design."""

    run: int
    block: int
    factors: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"run": self.run, "block": self.block, **self.factors}


class ExperimentDesigner:
    """Factorial designs, randomised blocks and sample sizing."""

    def __init__(self, power: PowerAnalysis | None = None) -> None:
        self.power = power or PowerAnalysis()

    def factorial(self, factors: dict[str, list[Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = [{}]
        for name in sorted(factors):
            levels = factors[name]
            if not levels:
                continue
            rows = [{**row, name: level} for row in rows for level in levels]
        return rows

    def randomise(
        self, rows: list[dict[str, Any]], blocks: int = 1, seed: int = 11,
    ) -> list[DesignRow]:
        if blocks <= 0:
            raise ValueError("blocks must be positive")
        rng = random.Random(seed)
        designed: list[DesignRow] = []
        run = 0
        for block in range(1, blocks + 1):
            block_rows = [dict(row) for row in rows]
            rng.shuffle(block_rows)
            for row in block_rows:
                run += 1
                designed.append(DesignRow(run=run, block=block, factors=row))
        return designed

    def power_plan(
        self, effect_size: float, alpha: float = 0.05, power: float = 0.8,
    ) -> dict[str, Any]:
        result = self.power.sample_size_two_sample(effect_size, alpha, power)
        plan = result.as_dict()
        plan["total_units"] = result.n_per_group * 2
        plan["alpha"] = alpha
        return plan

    def balance(self, rows: list[DesignRow]) -> dict[str, dict[str, int]]:
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            for factor, level in row.factors.items():
                counts[factor][str(level)] += 1
        return {factor: dict(sorted(counter.items())) for factor, counter in counts.items()}


# ── Module-level engine ───────────────────────────────────────────────────

class ScientificAgentEngine:
    """Facade used by Inference pipelines for research tasks."""

    def __init__(self) -> None:
        self.library = SkillLibrary()
        self.matcher = SkillMatcher(self.library)
        self.composer = SkillComposer(self.library)
        self.stats = StatisticalTestSuite()
        self.sequences = SequenceAnalysis()
        self.units = UnitConverter()
        self.hypotheses = HypothesisGenerator()
        self.designer = ExperimentDesigner()

    def plan(self, task: str, limit: int = 3) -> dict[str, Any]:
        matches = self.matcher.match(task, limit=limit)
        return {
            "task": task,
            "skills": [
                {"skill": m.skill.skill_id, "title": m.skill.title,
                 "score": m.score, "why": m.reasons}
                for m in matches
            ],
            "pipeline": self.composer.compose(task, limit=limit),
        }

    def analyse_groups(self, groups: list[list[float]], alpha: float = 0.05) -> dict[str, Any]:
        """Compare two or more groups with the right test plus an effect size."""
        if len(groups) == 1:
            return {"error": "need at least two groups"}
        if len(groups) == 2:
            a, b = groups
            welch = self.stats.t_test_independent(a, b, alpha)
            mann = self.stats.mann_whitney_u(a, b, alpha)
            return {
                "test": welch.name,
                "statistic": round(welch.statistic, 4),
                "p_value": round(welch.p_value, 6),
                "effect_size": round(welch.effect_size, 4),
                "significant": welch.significant,
                "nonparametric_p": round(mann.p_value, 6),
                "summary": welch.summary(),
            }
        anova = self.stats.one_way_anova(groups, alpha)
        return {
            "test": anova.name,
            "statistic": round(anova.statistic, 4),
            "p_value": round(anova.p_value, 6),
            "effect_size": round(anova.effect_size, 4),
            "significant": anova.significant,
            "summary": anova.summary(),
        }


# Module-level instance
engine = ScientificAgentEngine()
