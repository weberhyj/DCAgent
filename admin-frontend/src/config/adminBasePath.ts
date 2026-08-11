const DEFAULT_ADMIN_BASE_PATH = '/admin/'

function invalidBasePath(value: string): never {
  throw new Error(`Invalid VITE_ADMIN_BASE_PATH: ${JSON.stringify(value)}`)
}

export function normalizeAdminBasePath(value?: string): string {
  const candidate = value?.trim() || DEFAULT_ADMIN_BASE_PATH

  if (
    candidate.includes('\\')
    || candidate.includes('?')
    || candidate.includes('#')
    || candidate.startsWith('//')
    || /^[a-z][a-z\d+.-]*:/i.test(candidate)
  ) {
    return invalidBasePath(candidate)
  }

  if (candidate === '/') return '/'

  const segments = candidate.split('/').filter(Boolean)
  if (segments.length === 0 || segments.some(segment => segment === '.' || segment === '..')) {
    return invalidBasePath(candidate)
  }

  return `/${segments.join('/')}/`
}
