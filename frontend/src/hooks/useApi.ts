import { useCallback, useEffect, useState } from 'react'
import { request } from '../services/api'

export function useApi<T>(path: string | null) {
  const [state, setState] = useState<{
    key: string
    data: T | null
    error: string | null
  }>({ key: '', data: null, error: null })
  const [revision, setRevision] = useState(0)

  const reload = useCallback(() => setRevision((value) => value + 1), [])
  const key = `${path ?? ''}:${revision}`

  useEffect(() => {
    if (!path) return
    const controller = new AbortController()
    request<T>(path, { signal: controller.signal })
      .then((data) => setState({ key, data, error: null }))
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return
        setState({
          key,
          data: null,
          error: reason instanceof Error ? reason.message : 'Unable to load this data.',
        })
      })
    return () => controller.abort()
  }, [key, path])

  const current = state.key === key
  return {
    data: current ? state.data : null,
    loading: Boolean(path) && !current,
    error: current ? state.error : null,
    reload,
  }
}
