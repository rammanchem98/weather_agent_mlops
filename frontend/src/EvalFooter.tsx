import { useEffect, useState } from "react";
import { fetchEvalSummary, type EvalSummary } from "./api";

function formatDate(runAt: string): string {
  const date = new Date(runAt);
  return Number.isNaN(date.getTime()) ? runAt : date.toISOString().slice(0, 10);
}

// Shows the last recorded eval run, if the server has one. It describes a past
// run only; it makes no claim that evals run continuously. Hidden otherwise.
export default function EvalFooter({ load = fetchEvalSummary }: { load?: () => Promise<EvalSummary | null> }) {
  const [summary, setSummary] = useState<EvalSummary | null>(null);

  useEffect(() => {
    let active = true;
    load().then((result) => {
      if (active) setSummary(result);
    });
    return () => {
      active = false;
    };
  }, [load]);

  if (!summary) return null;
  return (
    <footer className="eval-footer">
      Eval harness: last run {formatDate(summary.runAt)} - {summary.passed} of {summary.total} golden queries passed
    </footer>
  );
}
