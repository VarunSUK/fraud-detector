// Load test for go-inference's real-time scoring path.
//
// Boot the stack first (see the README Quick Start), then:
//   k6 run tests/load-test.js
//   BASE_URL=http://staging.example.com k6 run tests/load-test.js
//
// This targets /api/v1/score, not /api/v1/decision -- decision writes to
// the (SQLite) audit log on every call, which isn't the thing to load-test
// with concurrent virtual users; see docs/deployment.md's note on the
// SQLite constraint. Scoring is the sub-100ms real-time path the README's
// performance claims are actually about.

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8080';

const errorRate = new Rate('errors');
const scoreLatency = new Trend('score_latency_ms', true);

export const options = {
  scenarios: {
    steady_load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '15s', target: 20 },
        { duration: '30s', target: 20 },
        { duration: '15s', target: 0 },
      ],
    },
  },
  thresholds: {
    // The README claims sub-100ms P95 for the rule-based path and ~50ms for
    // the ensemble; this threshold is intentionally looser (covers the
    // ml-serving HTTP hop) so a real run against the sidecar has a
    // realistic bar rather than the in-process-only rule-based number.
    http_req_duration: ['p(95)<500'],
    errors: ['rate<0.01'],
  },
};

function randomTransaction() {
  const isFraudLike = Math.random() < 0.1;
  const txn = {
    transaction_id: `loadtest_${__VU}_${__ITER}`,
    time: Math.floor(Math.random() * 172800),
    amount: isFraudLike ? 100 + Math.random() * 4000 : 1 + Math.random() * 300,
  };
  for (let i = 1; i <= 28; i++) {
    let value = randomGaussian();
    if (i === 14) value = isFraudLike ? randomGaussian(-2.6, 1.6) : randomGaussian();
    if (i === 4) value = isFraudLike ? randomGaussian(2.0, 1.6) : randomGaussian();
    txn[`v${i}`] = value;
  }
  return txn;
}

function randomGaussian(mean = 0, stdDev = 1) {
  const u1 = Math.random() || 1e-9;
  const u2 = Math.random();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + z * stdDev;
}

export default function () {
  const payload = JSON.stringify({ transaction: randomTransaction() });
  const params = { headers: { 'Content-Type': 'application/json' } };

  const res = http.post(`${BASE_URL}/api/v1/score`, payload, params);

  const ok = check(res, {
    'status is 200': (r) => r.status === 200,
    'has a score': (r) => {
      try {
        return typeof JSON.parse(r.body).score === 'number';
      } catch {
        return false;
      }
    },
  });

  errorRate.add(!ok);
  scoreLatency.add(res.timings.duration);

  sleep(0.1);
}
