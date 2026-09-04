# Deployment Guide

## Local: Docker Compose

```bash
docker-compose up -d
```

Services and how they depend on each other:

| Service | Depends on | Port |
|---|---|---|
| `zookeeper`, `kafka` | -- | 9092, 9101 |
| `redis` | -- | 6379 |
| `prometheus` | -- | 9090 |
| `grafana` | -- | 3000 (admin/admin) |
| `ml-training` | -- (runs once, trains and exits) | -- |
| `ml-serving` | `ml-training` | 8000 |
| `inference-api` | `ml-training`, `ml-serving` | 8080 |
| `frontend` | `inference-api` | 3001 (nginx, container listens on 8080) |
| `data-generator` | `kafka`, `redis` | -- |
| `kafka-ui` | `kafka` | 8081 |

Model artifacts live on the `ml_models` named volume (written by `ml-training`, read by both `ml-serving` and `inference-api`); the audit log lives on `audit_log` (written by `ml-serving`, mounted at `/app/audit`). `frontend`'s `VITE_API_BASE_URL` is baked in at image build time via a Docker build arg (see `docker-compose.yml`'s `frontend.build.args`) -- rebuild the image (`docker-compose build frontend`) if you change the inference API's externally-reachable URL.

`ml-training` runs `train.py` once against `creditcard.csv` and exits; it does **not** run `scripts/seed_audit_log.py`, so the audit log (and therefore the review queue / analytics dashboard) starts empty in this path. Run the seed script manually against the same volume if you want realistic history:
```bash
docker cp <ml-serving-container>:/app/models ./tmp-models  # or mount the same PVC/volume
python scripts/seed_audit_log.py --models-dir ./tmp-models --db ./tmp-audit.db
```
Simplest in practice: skip `ml-training`/`docker-compose` for model training, and instead run `python scripts/seed_audit_log.py --train` directly on the host into the same directories the containers mount (see the README Quick Start) -- it trains real models and seeds real history in one step.

## Kubernetes: Helm

```bash
helm dependency update helm/fraud-detection   # fetches Kafka/Redis/Prometheus/Grafana subcharts
helm install fraud-detection ./helm/fraud-detection \
  --set cloud.provider=aws \
  -f helm/fraud-detection/values-aws.yaml
```
Swap `values-aws.yaml` for `values-gcp.yaml` / `values-azure.yaml` as needed; each only overrides the cloud-specific IAM/storage annotations (see `_helpers.tpl`'s `cloudAnnotations`), not the workload definitions.

### What's deployed
`ml-training` (Job-like Deployment, run-to-completion pattern via `mlTraining.enabled`), `ml-serving`, `inference-api`, `frontend`, `data-generator`, plus the Kafka/Redis/Prometheus/Grafana subcharts declared in `Chart.yaml`. `inference-api`'s `ML_SERVICE_URL` env var is set from a template expression (`http://<release>-ml-serving:8000`), not a static value in `values.yaml`, so it resolves correctly regardless of release name.

### The SQLite constraint
`mlServing.replicaCount` defaults to **1** and should stay there. The audit log is SQLite on a `ReadWriteOnce` PVC (`ml-audit-pvc`) -- multiple `ml-serving` pods writing to the same SQLite file concurrently will corrupt it. Scaling `ml-serving` horizontally requires swapping the audit log for a real database (Postgres, etc.) first; `audit_log.py`'s functions (`record_decision`, `resolve_case`, `funnel_summary`, `score_decile_summary`) are the isolation boundary -- reimplementing them against a different backend shouldn't require touching `serve.py`'s routes.

### Security context and the frontend image
The chart applies a hardened pod-level `securityContext` (`runAsNonRoot`, `readOnlyRootFilesystem`, all capabilities dropped) to every container. This is why the frontend's production image is `nginxinc/nginx-unprivileged` rather than stock `nginx` -- stock nginx wants to bind port 80 as root and write to `/var/cache/nginx` as root, neither of which work under this securityContext. The unprivileged image listens on 8080 and, combined with `emptyDir` mounts for `/var/cache/nginx`, `/var/run`, and `/tmp` (declared in `templates/deployment.yaml`), runs correctly under the read-only root filesystem. The same reasoning is why `ml-training`, `ml-serving`, and `data-generator` each get a `/tmp` `emptyDir` -- Python's `tempfile` module and libraries like `shap`/`pandas` write there.

### Ingress
`inferenceAPI.ingress` and `frontend.ingress` are separate host entries (`fraud-detection-api.example.com`, `fraud-detection.example.com` by default) -- update both, and rebuild the frontend image with the matching `VITE_API_BASE_URL` build arg, before deploying to a real domain.

## CI

`.github/workflows/ci.yml` runs on every push/PR to `main`: Go build+vet+gofmt+test, Python pytest, frontend lint+test+build, and the analytics SQL regression tests. There is no CD (deploy-on-merge) pipeline in this repo -- the README's mention of "GitOps-ready with ArgoCD support" describes the Helm chart being structured in a way that *would* support GitOps tooling, not a configured ArgoCD Application in this repo.
