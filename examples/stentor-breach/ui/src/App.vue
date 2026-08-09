<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { getSnapshot, setControl } from './bridge'

const phases = ['CONTACT', 'PIERCE', 'RUPTURE', 'FEED', 'REPAIR']
const phase = ref(0)
const playing = ref(true)
const particleDensity = ref(0.25)
const membraneTension = ref(0.72)
const brownianMotion = ref(0.42)
const chemotaxis = ref(0.68)
const repairRate = ref(0.58)
const fps = ref(0)
const gpuMemory = ref(0)
const connected = ref(false)
let pollTimer: number | undefined

const phaseName = computed(() => phases[phase.value] ?? phases[0])
const particleCount = computed(() => Math.round(49600 * particleDensity.value))

function commit(name: string, value: number) {
  void setControl(name, value).catch(() => { connected.value = false })
}

function update(name: string, target: Event) {
  const value = Number((target.target as HTMLInputElement).value)
  if (name === 'interaction.membrane_tension') membraneTension.value = value
  else if (name === 'interaction.particle_density') particleDensity.value = value
  else if (name === 'interaction.brownian_motion') brownianMotion.value = value
  else if (name === 'interaction.chemotaxis') chemotaxis.value = value
  else if (name === 'interaction.repair_rate') repairRate.value = value
  commit(name, value)
}

function selectPhase(index: number) {
  phase.value = index
  playing.value = false
  commit('interaction.playing', 0)
  commit('interaction.phase', index)
}

function togglePlayback() {
  playing.value = !playing.value
  commit('interaction.playing', playing.value ? 1 : 0)
}

function reset() {
  commit('interaction.reset_scene', 1)
  window.setTimeout(() => commit('interaction.reset_scene', 0), 140)
}

async function pollSnapshot() {
  try {
    const snapshot = await getSnapshot()
    connected.value = snapshot.connected
    playing.value = (snapshot.values['interaction.playing'] ?? 1) > 0.5
    phase.value = Math.round(snapshot.values['interaction.phase'] ?? phase.value)
    particleDensity.value = snapshot.values['interaction.particle_density'] ?? particleDensity.value
    membraneTension.value = snapshot.values['interaction.membrane_tension'] ?? membraneTension.value
    brownianMotion.value = snapshot.values['interaction.brownian_motion'] ?? brownianMotion.value
    chemotaxis.value = snapshot.values['interaction.chemotaxis'] ?? chemotaxis.value
    repairRate.value = snapshot.values['interaction.repair_rate'] ?? repairRate.value
    fps.value = snapshot.values['interaction.hud_fps'] ?? fps.value
    gpuMemory.value = snapshot.values['interaction.hud_gpu_mb'] ?? gpuMemory.value
  } catch { connected.value = false }
}

onMounted(() => {
  void pollSnapshot()
  pollTimer = window.setInterval(pollSnapshot, 250)
})

onBeforeUnmount(() => {
  if (pollTimer !== undefined) window.clearInterval(pollTimer)
})
</script>

<template>
  <main>
    <header>
      <div>
        <h1>MEMBRANE <i>/</i> 01</h1>
        <p>STENTOR <b>×</b> HOLOPHRYA</p>
        <span>A living point-cloud study</span>
      </div>
      <button class="transport" :aria-label="playing ? 'Pause' : 'Play'" @click="togglePlayback">
        <svg v-if="playing" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h3v14H7zm7 0h3v14h-3z" /></svg>
        <svg v-else viewBox="0 0 24 24" aria-hidden="true"><path d="m8 5 11 7-11 7z" /></svg>
      </button>
    </header>

    <section class="status-line">
      <span :class="{ live: connected }"><i></i>{{ connected ? 'LIVE' : 'WAITING' }}</span>
      <span>{{ (particleCount / 1000).toFixed(1) }}K PARTICLES</span>
      <span>{{ Math.round(fps) }} FPS</span>
    </section>

    <section class="phase-block">
      <div class="section-title"><span>CYCLE</span><output>{{ phaseName }}</output></div>
      <div class="timeline" aria-label="Simulation phase">
        <button v-for="(name, index) in phases" :key="name" :class="{ active: phase === index && !playing }" @click="selectPhase(index)">
          <i></i><span>{{ name }}</span>
        </button>
      </div>
      <p>Select a phase to pause and inspect the specimen.</p>
    </section>

    <section class="controls">
      <div class="section-title"><span>BIOPHYSICS</span><output>µm · s⁻¹</output></div>

      <label>
        <span>Particle population <output>{{ particleCount.toLocaleString() }}</output></span>
        <input type="range" min="0.16" max="1" step="0.01" :value="particleDensity" @input="update('interaction.particle_density', $event)" />
        <small>Visible samples across all cell structures</small>
      </label>

      <label>
        <span>Membrane tension <output>{{ membraneTension.toFixed(2) }}</output></span>
        <input type="range" min="0" max="1" step="0.01" :value="membraneTension" @input="update('interaction.membrane_tension', $event)" />
        <small>Elastic restoration of the Stentor shell</small>
      </label>
      <label>
        <span>Brownian motion <output>{{ brownianMotion.toFixed(2) }}</output></span>
        <input type="range" min="0" max="1" step="0.01" :value="brownianMotion" @input="update('interaction.brownian_motion', $event)" />
        <small>Thermal forcing inside the aqueous medium</small>
      </label>
      <label>
        <span>Chemotaxis <output>{{ chemotaxis.toFixed(2) }}</output></span>
        <input type="range" min="0" max="1" step="0.01" :value="chemotaxis" @input="update('interaction.chemotaxis', $event)" />
        <small>Holophrya response to the nutrient plume</small>
      </label>
      <label>
        <span>Repair rate <output>{{ repairRate.toFixed(2) }}</output></span>
        <input type="range" min="0" max="1" step="0.01" :value="repairRate" @input="update('interaction.repair_rate', $event)" />
        <small>Wound contraction and material recovery</small>
      </label>
    </section>

    <footer>
      <button class="reset" @click="reset">RESET SPECIMEN</button>
      <span>{{ Math.round(gpuMemory) }} MiB GPU</span>
    </footer>
  </main>
</template>
