interface SimpleCardProps {
  title: string;
  value: number | string;
}

export default function SimpleCard({ title, value }: SimpleCardProps) {
  return (
    <div className="border rounded-lg p-4 shadow-sm max-w-md">
      <h2 className="text-xl font-semibold">{title}</h2>
      <p>Value: {value}</p>
    </div>
  );
}
