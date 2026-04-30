/**
 * computeRougeL — Compute ROUGE-L F1 score between candidate and reference strings.
 * Pure JS implementation (no dependencies). Works on word-level tokens.
 */

function lcsLength(a, b) {
  const m = a.length, n = b.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (a[i - 1] === b[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
  }
  return dp[m][n];
}

export function computeRougeL(candidate, reference) {
  if (!candidate || !reference) return null;
  const candTokens = candidate.toLowerCase().trim().split(/\s+/);
  const refTokens = reference.toLowerCase().trim().split(/\s+/);

  if (candTokens.length === 0 || refTokens.length === 0) return null;

  const lcs = lcsLength(candTokens, refTokens);

  const precision = lcs / candTokens.length;
  const recall = lcs / refTokens.length;

  if (precision + recall === 0) return 0;

  const f1 = (2 * precision * recall) / (precision + recall);
  return Math.round(f1 * 10000) / 10000; // 4 decimal places
}

/**
 * Compute multiple metrics for a candidate against a reference.
 */
export function computeMetrics(candidate, reference) {
  if (!candidate || !reference) return null;

  const rougeL = computeRougeL(candidate, reference);

  // Tokenization (mirrors evaluate.py _tokenize)
  const tokenize = (text) => text.toLowerCase().replace(/[^a-z0-9\s]/g, '').split(/\s+/).filter(Boolean);
  const hyp = tokenize(candidate);
  const ref = tokenize(reference);

  if (ref.length === 0) return { rougeL, overlap: 0 };

  // Unique tokens for overlap recall
  const refSet = new Set(ref);
  const hypSet = new Set(hyp);
  const common = [...refSet].filter(w => hypSet.has(w)).length;
  const overlap = Math.round((common / refSet.size) * 1000) / 10; // e.g. 14.5

  return { rougeL, overlap };
}
