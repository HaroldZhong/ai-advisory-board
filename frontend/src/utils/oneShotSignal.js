export function shouldConsumeOneShotSignal(signal, lastConsumedSignal) {
  return signal > 0 && signal !== lastConsumedSignal;
}
