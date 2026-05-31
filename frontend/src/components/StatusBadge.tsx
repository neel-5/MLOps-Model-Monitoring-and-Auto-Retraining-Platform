type StatusBadgeProps = {
  status: string;
  tone?: "ok" | "warn" | "danger" | "neutral";
};

export function StatusBadge({ status, tone = "neutral" }: StatusBadgeProps) {
  return <span className={`status-badge status-${tone}`}>{status}</span>;
}
