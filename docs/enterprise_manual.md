# QuantumCloud Enterprise Infrastructure, Security & Operations Manual
**Document Version:** 4.2.0 | **Classification:** Internal Engineering Confidential | **Last Updated:** 2026-09-15

---

## 1. Cloud Infrastructure & Kubernetes Cluster Orchestration

QuantumCloud operates across three multi-region AWS availability zones (us-east-1, us-west-2, eu-central-1). Our containerized microservices fleet runs on Amazon EKS (Elastic Kubernetes Service) with Kubernetes version 1.30.

### 1.1 Node Pool Specifications & Autoscaling Limits
- **Core Processing Pool (`pool-compute-c6i`):**
  - Instance Type: `c6i.4xlarge` (16 vCPUs, 32 GiB memory).
  - Autoscaler trigger threshold: 75% sustained CPU utilization over a 3-minute window.
  - Min replicas: 6 nodes, Max replicas: 48 nodes per cluster zone.
- **Memory-Optimized Cache Pool (`pool-cache-r6i`):**
  - Instance Type: `r6i.2xlarge` (8 vCPUs, 64 GiB ECC memory).
  - Min replicas: 4 nodes, Max replicas: 16 nodes.

### 1.2 Ingress Gateway & Service Mesh
All incoming external internet traffic terminates at AWS Network Load Balancers (NLB) before forwarding to the Envoy-based Ingress Controller.
- Internal service-to-service communication is governed by Istio 1.22 in ambient mesh mode with automatic mutual TLS (mTLS) enforcement using 2048-bit RSA ephemeral certificates rotated every 24 hours.
- Distributed tracing is powered by OpenTelemetry collector sidecars exporting trace spans to Honeycomb.

---

## 2. Cyber Security, Identity & Incident Response Protocols

QuantumCloud enforces a strict Zero-Trust Architecture (ZTA) across all microservice ingress boundaries and management APIs.

### 2.1 OAuth 2.0 & STS Ephemeral Token Lifecycle
All API clients, automated worker agents, and internal microservices must obtain a cryptographically signed OAuth 2.0 JSON Web Token (JWT) issued by the centralized Security Token Service (STS).
- **Token Validity Window:** Ephemeral expiration window of exactly 900 seconds (15 minutes).
- **Required Claims:**
  - `iss`: `https://sts.quantumcloud.internal/v2`
  - `aud`: `urn:quantum:api`
  - `tenant_id`: RFC-4122 compliant UUID v4 string.
  - `scope`: Least-privilege role matrix (e.g. `read:analytics`, `write:orders`).

### 2.2 Critical Incident Playbook: ERR-9021-TOKEN-REVOKED
When the STS anomaly engine detects anomalous token usage, compromised credentials, or replay attack patterns, it emits an emergency event `ERR-9021-TOKEN-REVOKED`. Engineers on on-call escalation must execute the following remediation sequence:

1. **Purge Redis Client Session:**
   ```bash
   redis-cli -h redis-auth.internal -p 6379 -a "$REDIS_AUTH_TOKEN" DEL "session:$SESSION_ID"
   ```
2. **Rotate Vault Secret Accessor:**
   ```bash
   vault write auth/approle/role/gateway-prod/secret-id-accessor/destroy accessor=$ACCESSOR_ID
   ```
3. **Escalate & Audit Notification:**
   - Fire a PagerDuty high-urgency incident notification to the SecOps On-Call schedule.
   - Initiate an automated AWS CloudTrail query:
     ```bash
     aws cloudtrail lookup-events --lookup-attributes AttributeKey=Username,AttributeValue="$CLIENT_ID" --max-results 50
     ```

### 2.3 Distributed Rate Limiting (ERR-4290)
All inbound requests pass through Envoy proxies with token bucket limiters:
- **Standard Tier:** 5,000 requests/minute with a burst allowance of 250 requests.
- **Enterprise Tier:** 50,000 requests/minute with a burst allowance of 2,500 requests.
- When limits are exceeded, HTTP 429 Too Many Requests is returned with `Retry-After: <seconds>`.

---

## 3. Distributed Database, Multi-Tenant Partitioning & Indexing

Our primary transactional data tier is powered by a high-availability PostgreSQL 16 cluster with physical streaming replication across 12 read replicas.

### 3.1 Composite Indexing Guidelines
To prevent sequential table scans and eliminate deadlocks across multi-tenant tables:
- **Rule 1 (Leading Key):** All composite B-Tree indexes must lead with `tenant_id` followed by monotonically increasing timestamp `created_at`.
- **Rule 2 (Partial Indexes):** High-frequency transactional filtering requires filtered partial indexes:
  ```sql
  CREATE INDEX CONCURRENTLY idx_active_orders
  ON orders(tenant_id, created_at)
  WHERE status = 'ACTIVE' AND deleted_at IS NULL;
  ```
- **Rule 3 (Index Bloat Maintenance):** Tables exceeding 100 million rows run autovacuum with `autovacuum_vacuum_scale_factor = 0.05` and `autovacuum_max_workers = 6`.

### 3.2 PgBouncer Connection Pooling
- Microservice pods connect via PgBouncer running in **transaction pooling mode**.
- Max server connection pool per microservice replica pod: strictly **20 connections**.
- Client query timeout ceiling: **4,500 milliseconds**.

---

## 4. Distributed Cache Eviction & Event-Driven Architecture

The distributed caching subsystem utilizes Redis 7.2 / Valkey running in an active-active cluster configuration across availability zones.

### 4.1 Cache TTL & Eviction Policies
- Default Cache TTL: **3,600 seconds (1 hour)** for standard query responses.
- Eviction Algorithm: `volatile-lru` (Least Recently Used with explicit TTL tags).
- Memory Cap: 48 GiB per Redis shard with auto-snapshotting to Amazon S3 every 6 hours.

### 4.2 Kafka Change Data Capture (CDC) Eviction
When PostgreSQL writes or updates occur, Debezium CDC captures the mutation and produces an event to Apache Kafka topic `db.mutations.v1`.
- Microservice pods consume the Kafka topic to evict local in-memory LRU cache entries, ensuring global eventual consistency within **25 milliseconds**.

---

## 5. AI Gateway, Semantic Prompt Caching & Model Fallback

The QuantumCloud AI Platform routes prompt workloads across multiple LLM providers via an internal routing proxy.

### 5.1 Dynamic Complexity Routing
- **Low-Complexity Tasks (Classification, Metadata Tagging):**
  - Routed to Thinking Machines: Inkling / Google Gemma 2 / Llama-3.1-8B with sub-180ms Time-To-First-Token (TTFT).
- **High-Complexity Tasks (Code Synthesis, Multi-Hop Architectural Analysis):**
  - Routed to frontier models (Claude 3.5 Sonnet / GPT-4o) with automatic fallback to OpenRouter when provider 5xx rates exceed 2%.

### 5.2 ChromaDB Semantic Prompt Cache
To reduce GPU inference costs and slash latency to < 15ms:
- Prompts are embedded using local ONNX `all-MiniLM-L6-v2` embeddings (384 dimensions).
- **Cache Hit Threshold:** If incoming prompt cosine similarity is **>= 0.94**, the cached LLM response is returned immediately.
- **Cache Expiration:** Semantic cache entries expire automatically after **24 hours**.

---

## 6. Employee Travel, Expense & Meal Reimbursement Policy *(HR Guidelines)*

*Note: This section covers corporate operations and does not relate to cloud infrastructure.*

### 6.1 Per Diem & Meal Expense Guidelines
- Domestic Travel Per Diem: Up to **$75.00 USD per day** for breakfast ($15), lunch ($25), and dinner ($35).
- International Travel Per Diem: Up to **$110.00 USD per day**.
- Alcoholic beverages are non-reimbursable unless part of an authorized client entertainment dinner approved in advance by a VP.

### 6.2 Ground Transportation & Ridesharing
- Employees must use Uber for Business or Lyft Corporate linked to the company expense account.
- Rental cars require prior manager approval and must be booked in the "Intermediate" vehicle class.

---

## 7. Office Culinary Guild & Coffee Equipment Calibration Guide *(Recreational)*

*Note: This section is for office recreation only.*

### 7.1 Artisanal Sourdough Fermentation
- **Levain Ratio:** Refresh wild sourdough starter at a **1:2:2 ratio** (starter : water : bread flour) 12 hours prior to final mixing.
- **Bulk Fermentation Ambient Temperature:** Maintain precisely **26°C (78°F)** for 4.5 hours with hourly stretch-and-folds.
- **Bake Specs:** Preheat Dutch oven to **230°C (450°F)** with steam injection for the first 20 minutes.

### 7.2 La Marzocco Espresso Machine Calibration
- **Dose:** 18.0 grams of freshly ground single-origin Ethiopian espresso beans.
- **Yield:** 36.0 grams liquid espresso extraction in 28 to 32 seconds at **9.0 bars of pump pressure** and **93.5°C water temperature**.
