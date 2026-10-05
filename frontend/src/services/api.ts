const API_BASE = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...init?.headers,
      },
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Network request failed.'
    throw new Error(`Cannot reach the UdyamPulse API. ${message}`)
  }

  const payload: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    const detail =
      typeof payload === 'object' && payload !== null && 'detail' in payload
        ? String(payload.detail)
        : `Request failed with status ${response.status}.`
    throw new Error(detail)
  }
  return payload as T
}

export function downloadCsv<T extends object>(filename: string, rows: T[]) {
  if (!rows.length) return
  const records = rows.map((row) => new Map(Object.entries(row)))
  const columns = Array.from(records[0].keys())
  const escape = (value: unknown) => {
    const text = value == null ? '' : String(value)
    return `"${text.replaceAll('"', '""')}"`
  }
  const content = [
    columns.map(escape).join(','),
    ...records.map((row) => columns.map((column) => escape(row.get(column))).join(',')),
  ].join('\r\n')
  const url = URL.createObjectURL(new Blob([content], { type: 'text/csv;charset=utf-8' }))
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}
