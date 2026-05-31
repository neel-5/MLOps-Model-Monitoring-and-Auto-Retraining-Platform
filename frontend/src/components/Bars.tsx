type BarItem = {
  label: string;
  value: number;
  tone?: "teal" | "amber" | "rose" | "indigo";
};

export function Bars({ items, maxValue }: { items: BarItem[]; maxValue?: number }) {
  const max = maxValue ?? Math.max(...items.map((item) => item.value), 0.01);
  return (
    <div className="bars">
      {items.map((item) => (
        <div className="bar-row" key={item.label}>
          <span title={item.label}>{item.label}</span>
          <div className="bar-track">
            <i className={`bar-fill tone-${item.tone ?? "teal"}`} style={{ width: `${Math.min((item.value / max) * 100, 100)}%` }} />
          </div>
          <b>{item.value.toFixed(item.value < 1 ? 3 : 0)}</b>
        </div>
      ))}
    </div>
  );
}
