import { invoke } from '@tauri-apps/api/core'

export interface PanelSnapshot {
  connected: boolean
  values: Record<string, number>
}

export function getSnapshot(): Promise<PanelSnapshot> {
  return invoke<PanelSnapshot>('get_snapshot')
}

export function setControl(name: string, value: number): Promise<void> {
  return invoke<void>('set_control', { name, value })
}
