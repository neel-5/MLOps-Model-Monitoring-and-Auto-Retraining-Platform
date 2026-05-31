import type { ConfusionMatrix as ConfusionMatrixType } from "../types";

export function ConfusionMatrix({ data }: { data?: ConfusionMatrixType }) {
  if (!data || !data.matrix?.length) {
    return <div className="empty-state">No confusion matrix yet.</div>;
  }

  return (
    <div className="matrix-wrap">
      <table className="matrix">
        <thead>
          <tr>
            <th>Actual</th>
            {data.labels.map((label) => (
              <th key={`pred-${label}`}>Pred {label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.matrix.map((row, rowIndex) => (
            <tr key={`row-${data.labels[rowIndex]}`}>
              <th>{data.labels[rowIndex]}</th>
              {row.map((value, columnIndex) => (
                <td key={`${rowIndex}-${columnIndex}`} className={rowIndex === columnIndex ? "matrix-hit" : ""}>
                  {value}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
