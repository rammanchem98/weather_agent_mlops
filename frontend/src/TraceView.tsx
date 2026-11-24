import type { Trace } from "./api";

function formatArgs(args: Record<string, unknown>): string {
  return Object.entries(args)
    .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
    .join(", ");
}

function seconds(value: number | undefined): string | null {
  return value === undefined ? null : `${value.toFixed(2)}s`;
}

// "ok" only when every layer that ran passed; "unavailable" is shown as a warning.
function guardrailTone(guardrail: Record<string, string>): "ok" | "warn" | "blocked" {
  const statuses = Object.values(guardrail);
  if (statuses.includes("blocked")) return "blocked";
  if (statuses.some((s) => s !== "pass" && s !== "passed")) return "warn";
  return "ok";
}

function GuardrailBadge({ guardrail }: { guardrail: Record<string, string> }) {
  const layers = Object.entries(guardrail);
  if (layers.length === 0) return null;
  return (
    <span className={`badge badge-${guardrailTone(guardrail)}`}>
      Guardrail: {layers.map(([layer, status]) => `${layer} ${status}`).join(" · ")}
    </span>
  );
}

export default function TraceView({ trace }: { trace: Trace }) {
  const meta = [
    trace.rounds !== undefined ? `${trace.rounds} tool round${trace.rounds === 1 ? "" : "s"}` : null,
    trace.capped ? "stopped by the tool-call limit" : null,
    trace.geminiCalls !== undefined ? `${trace.geminiCalls} Gemini call${trace.geminiCalls === 1 ? "" : "s"}` : null,
    trace.totalLatencyS !== undefined ? `${seconds(trace.totalLatencyS)} total` : null,
  ].filter(Boolean);

  return (
    <details className="trace" open>
      <summary>How I got this answer</summary>
      <div className="trace-body">
        <GuardrailBadge guardrail={trace.guardrail} />

        {trace.plan && (
          <div className="trace-section">
            <h3>Reasoning summary</h3>
            <p className="trace-plan">{trace.plan}</p>
          </div>
        )}

        <div className="trace-section">
          <h3>Tool calls</h3>
          {trace.toolCalls.length === 0 ? (
            <p className="trace-muted">No tools were called.</p>
          ) : (
            <ol className="trace-tools">
              {trace.toolCalls.map((call, index) => (
                <li key={index}>
                  <code>
                    {call.name}({formatArgs(call.args)})
                  </code>
                  {seconds(call.durationS) && <span className="trace-muted"> · {seconds(call.durationS)}</span>}
                  {call.resultPreview && <div className="trace-preview">{call.resultPreview}</div>}
                </li>
              ))}
            </ol>
          )}
        </div>

        {meta.length > 0 && <p className="trace-muted trace-meta">{meta.join(" · ")}</p>}
      </div>
    </details>
  );
}
