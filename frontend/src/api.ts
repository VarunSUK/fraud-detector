import type {
  AccountContext,
  AnalyticsSummary,
  ApiError,
  CasesResponse,
  DecisionResponse,
  HealthResponse,
  Transaction,
} from "./types";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8080";

export class ApiRequestError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
    this.name = "ApiRequestError";
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!resp.ok) {
    let message = `Request failed with status ${resp.status}`;
    try {
      const body = (await resp.json()) as ApiError;
      message = body.message || body.error || message;
    } catch {
      // response body wasn't JSON; keep the generic message
    }
    throw new ApiRequestError(message, resp.status);
  }

  return resp.json() as Promise<T>;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export function getDecision(
  transaction: Transaction,
  account: AccountContext,
): Promise<DecisionResponse> {
  return request<DecisionResponse>("/api/v1/decision", {
    method: "POST",
    body: JSON.stringify({ transaction, account }),
  });
}

export function getCases(): Promise<CasesResponse> {
  return request<CasesResponse>("/api/v1/cases");
}

export function resolveCase(
  caseId: number,
  verdict: "approve" | "decline",
  isActualFraud?: boolean,
): Promise<{ case_id: number; verdict: string }> {
  return request(`/api/v1/cases/${caseId}/resolve`, {
    method: "POST",
    body: JSON.stringify({ verdict, is_actual_fraud: isActualFraud }),
  });
}

export function getAnalyticsSummary(): Promise<AnalyticsSummary> {
  return request<AnalyticsSummary>("/api/v1/analytics/summary");
}
