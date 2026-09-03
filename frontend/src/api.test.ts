import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiRequestError, getCases, getHealth } from "./api";

describe("api client", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("resolves with parsed JSON on success", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ status: "healthy", timestamp: "now", version: "1.0.0", models: [] }),
      }),
    );

    const health = await getHealth();
    expect(health.status).toBe("healthy");
  });

  it("throws ApiRequestError with the server message on failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 503,
        json: async () => ({ error: "Decisioning unavailable", message: "sidecar down" }),
      }),
    );

    await expect(getCases()).rejects.toMatchObject(
      new ApiRequestError("sidecar down", 503),
    );
  });

  it("falls back to a generic message when the error body isn't JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        json: async () => {
          throw new Error("not json");
        },
      }),
    );

    await expect(getCases()).rejects.toThrow("Request failed with status 500");
  });
});
