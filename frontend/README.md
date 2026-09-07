# Fraud & Credit Risk Console

React + TypeScript + Vite frontend for the [fraud detection system](../README.md). Three tabs against the live go-inference API:

- **Decision Console** -- score a transaction (with sample-data generators for a "normal" or "fraud-like" transaction), see the resulting action, credit-limit recommendation, narrative, per-model score breakdown, and a signed SHAP contribution chart.
- **Review Queue** -- the human-in-the-loop case queue: transactions routed to `step_up_review`, with approve/decline actions that call `POST /api/v1/cases/:id/resolve`.
- **Analytics** -- a live approval funnel and fraud-rate-by-score-decile chart, pulled from `GET /api/v1/analytics/summary`.

## Running it

```bash
npm install
npm run dev       # dev server on :5173, proxies nothing -- talks straight to VITE_API_BASE_URL
npm run build     # type-checks (tsc -b) then builds to dist/
npm run test      # vitest
npm run lint      # eslint
```

`VITE_API_BASE_URL` (default `http://localhost:8080`) is read at **build time** -- Vite inlines `import.meta.env.VITE_*` vars into the bundle, so changing it after building does nothing. Set it before `npm run dev`/`npm run build`, or via the Docker build arg (see `Dockerfile`). The go-inference API needs to actually be running (with a healthy `ml-serving` sidecar behind it) for anything past the health indicator dot to work -- see the [README Quick Start](../README.md#-quick-start) for the full local sequence.

## Structure

```
src/
  api.ts                     # typed fetch wrappers for every go-inference endpoint
  types.ts                   # response/request shapes, mirrored from go-inference's models package
  sampleData.ts              # generates plausible "normal" / "fraud-like" sample transactions client-side
  components/
    TransactionForm.tsx      # the scoring form + account context inputs
    DecisionResult.tsx       # score, action/risk badges, narrative, credit-limit delta, model breakdown
    ShapChart.tsx             # diverging bar chart for signed SHAP contributions
    ActionBadge.tsx          # approve/step_up_review/decline + risk tier styling
    ReviewQueue.tsx          # pending-cases list + resolve actions
    AnalyticsDashboard.tsx   # funnel + decile chart
```

## Why nginx-unprivileged in production

The Docker production stage uses `nginxinc/nginx-unprivileged` rather than stock `nginx`, listening on 8080. That's not a style choice -- the Helm chart runs every pod under a hardened `securityContext` (`runAsNonRoot`, `readOnlyRootFilesystem`), and stock nginx wants to bind port 80 as root and write to `/var/cache/nginx` as root. See [docs/deployment.md](../docs/deployment.md#security-context-and-the-frontend-image) for the full reasoning.
