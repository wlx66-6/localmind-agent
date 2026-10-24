/** Server timestamps are UTC ISO; show the user's own calendar date. */
export function localDate(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value.slice(0, 10)
  return date.toLocaleDateString('sv-SE')
}