import { computed, onScopeDispose, shallowRef } from 'vue'
import { fetchBackendVersion } from '@/services/api'

export function useBackendVersion() {
  const version = shallowRef<string | null>(null)
  let active = true

  onScopeDispose(() => {
    active = false
  })

  async function load() {
    try {
      const loadedVersion = await fetchBackendVersion()
      if (active) version.value = loadedVersion
    }
    catch {
      if (active) version.value = null
    }
  }

  const displayVersion = computed(() => (
    version.value ? `v${version.value}` : '版本未知'
  ))

  return { displayVersion, load }
}
