export type Layout = 'manhattan' | 'barcelona' | 'tokyo'
export type ColorMode = 'emotion' | 'mode'

export const POPULATION = 50_000

export interface AgentField {
  lane: Uint16Array
  cross: Float32Array
  phase: Float32Array
  speed: Float32Array
  emotion: Uint8Array
  mode: Uint8Array
}

function splitmix(value: number) {
  let z = (value + 0x9e3779b9) >>> 0
  z = Math.imul(z ^ (z >>> 16), 0x21f0aaad)
  z = Math.imul(z ^ (z >>> 15), 0x735a2d97)
  return ((z ^ (z >>> 15)) >>> 0) / 4294967296
}

// All stochasticity is confined to initialization. Animation and decisions are deterministic.
export function initializeAgents(seed = 18472): AgentField {
  const field: AgentField = {
    lane: new Uint16Array(POPULATION),
    cross: new Float32Array(POPULATION),
    phase: new Float32Array(POPULATION),
    speed: new Float32Array(POPULATION),
    emotion: new Uint8Array(POPULATION),
    mode: new Uint8Array(POPULATION),
  }
  for (let i = 0; i < POPULATION; i += 1) {
    field.lane[i] = Math.floor(splitmix(seed + i * 7) * 22)
    field.cross[i] = splitmix(seed + i * 13) * 2 - 1
    field.phase[i] = splitmix(seed + i * 19)
    field.mode[i] = Math.floor(splitmix(seed + i * 23) * 5)
    field.emotion[i] = Math.floor(splitmix(seed + i * 29) * 5)
    field.speed[i] = 0.45 + splitmix(seed + i * 31) * 1.5
  }
  return field
}

export const AFFECT_COLORS = ['#2196f3', '#45dc69', '#ffc21c', '#ff552d', '#9d50cf']
export const MODE_COLORS = ['#f4f0e7', '#6fe6ff', '#ffc21c', '#b97af5', '#ff6d50']

export function agentDetail(id: number, minute: number) {
  const intents = ['Work', 'Education', 'Care', 'Errand', 'Social']
  const modes = ['Walk → Train', 'Walk', 'Car', 'Cycle', 'Walk → Air']
  const confidence = 82 + ((id * 11 + minute) % 17)
  return {
    id: `A-${String(id).padStart(5, '0')}`,
    intent: intents[id % intents.length],
    mode: modes[(id * 3) % modes.length],
    next: `00:${String(8 + ((id + minute) % 19)).padStart(2, '0')}`,
    confidence,
  }
}
