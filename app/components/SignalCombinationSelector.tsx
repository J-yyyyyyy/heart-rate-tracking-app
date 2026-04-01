'use client';

export type SignalCombinationMode = 'default' | 'redOnly' | 'greenOnly' | 'blueOnly' | '2xG-R-B';

interface Props {
  value: SignalCombinationMode;
  onChange: (mode: SignalCombinationMode) => void;
}

const SIGNAL_COMBINATION_OPTIONS: { value: SignalCombinationMode; label: string }[] = [
  { value: 'default', label: 'Default (2R−G−B)' },
  { value: 'redOnly', label: 'Red only' },
  { value: 'greenOnly', label: 'Green only' },
  { value: 'blueOnly', label: 'Blue only' },
  { value: '2xG-R-B', label: '2×G−R−B' },
];

export default function SignalCombinationSelector({ value, onChange }: Props) {
  return (
    <div className="w-56">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as SignalCombinationMode)}
        className="w-full px-4 py-3 text-base font-medium text-gray-900 bg-white border-2 border-gray-300 rounded-xl shadow-md cursor-pointer hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-all"
      >
        {SIGNAL_COMBINATION_OPTIONS.map((option) => (
          <option key={option.value} value={option.value} className="text-gray-900 bg-white">
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}
