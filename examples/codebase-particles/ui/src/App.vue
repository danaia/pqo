<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { getSnapshot, setControl } from './bridge'
import { FILES } from './files'

const connected = ref(false)
const selectedId = ref(0)
const zoom = ref(1)
const orbitAngle = ref(0)
const orbitSelected = ref(false)
const fps = ref(0)
const gpuMemory = ref(0)
let pollTimer: number | undefined

const selectedFile = computed(() => FILES[selectedId.value] ?? 'Unknown file')
const selectedGroup = computed(() => {
  if (selectedId.value < 44) return 'Compiler & runtime'
  if (selectedId.value < 78) return 'Documentation'
  if (selectedId.value < 120) return 'Examples'
  if (selectedId.value < 150) return 'Baseline application'
  return 'Support & root'
})

function commit(name: string, value: number) {
  void setControl(name, value).catch(() => { connected.value = false })
}

function toggleOrbit() {
  orbitSelected.value = !orbitSelected.value
  commit('interaction.orbit_selected', orbitSelected.value ? 1 : 0)
}

function updateOrbit(event: Event) {
  orbitAngle.value = Number((event.target as HTMLInputElement).value)
  commit('camera.orbit_angle', orbitAngle.value)
}

function updateZoom(event: Event) {
  zoom.value = Number((event.target as HTMLInputElement).value)
  commit('camera.zoom', zoom.value)
}

function resetCamera() {
  zoom.value = 1
  orbitAngle.value = 0
  orbitSelected.value = false
  commit('camera.reset', 1)
}

async function pollSnapshot() {
  try {
    const snapshot = await getSnapshot()
    connected.value = snapshot.connected
    selectedId.value = Math.max(0, Math.min(FILES.length - 1,
      Math.round(snapshot.values['interaction.selected'] ?? selectedId.value)))
    zoom.value = snapshot.values['camera.zoom'] ?? zoom.value
    orbitAngle.value = snapshot.values['camera.orbit_angle'] ?? orbitAngle.value
    orbitSelected.value = (snapshot.values['camera.orbit_selected'] ?? 0) > 0.5
    fps.value = snapshot.values['interaction.hud_fps'] ?? fps.value
    gpuMemory.value = snapshot.values['interaction.hud_gpu_mb'] ?? gpuMemory.value
  } catch {
    connected.value = false
  }
}

onMounted(() => {
  void pollSnapshot()
  pollTimer = window.setInterval(pollSnapshot, 100)
})

onBeforeUnmount(() => {
  if (pollTimer !== undefined) window.clearInterval(pollTimer)
})
</script>

<template>
  <main>
    <header class="app-header">
      <div>
        <p class="eyebrow">PQO REPOSITORY</p>
        <h1>Particle Inspector</h1>
      </div>
      <span class="status" :class="{ live: connected }">{{ connected ? 'Live' : 'Waiting' }}</span>
    </header>

    <section class="selection">
      <p class="eyebrow">SELECTED FILE · #{{ selectedId }}</p>
      <h2>{{ selectedFile }}</h2>
      <p>{{ selectedGroup }}</p>
    </section>

    <section class="instructions">
      <div><kbd>Click</kbd><span>Select a particle</span></div>
      <div><kbd>Scroll</kbd><span>Zoom at mouse location</span></div>
      <div><kbd>Drag</kbd><span>Orbit around selection</span></div>
    </section>

    <section class="controls">
      <button :class="{ active: orbitSelected }" @click="toggleOrbit">
        <span class="orbit-icon">◎</span>
        {{ orbitSelected ? 'Orbiting selection' : 'Orbit selection' }}
      </button>

      <label>
        <span><b>Orbit angle</b><output>{{ orbitAngle.toFixed(2) }} rad</output></span>
        <input type="range" min="-3.1416" max="3.1416" step="0.01" :value="orbitAngle" @input="updateOrbit" />
      </label>

      <label>
        <span><b>Zoom</b><output>{{ zoom.toFixed(2) }}×</output></span>
        <input type="range" min="0.55" max="6" step="0.05" :value="zoom" @input="updateZoom" />
      </label>

      <button class="secondary" @click="resetCamera">Reset camera</button>
    </section>

    <footer>
      <span>{{ Math.round(fps) }} FPS</span>
      <span>{{ Math.round(gpuMemory) }} MiB GPU</span>
      <span>{{ FILES.length }} files</span>
    </footer>
  </main>
</template>
